
#include "zkx/maybe_owning.h"

#include "gtest/gtest.h"

namespace zkx {

TEST(MaybeOwningTest, Null) {
  MaybeOwning<char> m(nullptr);
  EXPECT_EQ(m.get(), nullptr);
  EXPECT_EQ(m.get_mutable(), nullptr);
}

TEST(MaybeOwningTest, Owning) {
  MaybeOwning<char> m(std::make_unique<char>());
  *m.get_mutable() = 'a';
  EXPECT_EQ(*m, 'a');
}

TEST(MaybeOwningTest, Shared) {
  auto owner = std::make_unique<char>();
  *owner = 'x';
  MaybeOwning<char> c1(owner.get());
  MaybeOwning<char> c2(owner.get());

  EXPECT_EQ(*c1, 'x');
  EXPECT_EQ(*c2, 'x');
  EXPECT_EQ(c1.get(), c2.get());
}

}  // namespace zkx
