//
// Created by samuel on 01/07/2026.
//

#include <arenai_model/item.h>
#include <arenai_model_tests/tests_item/tests_life_item.h>

// ========================================================================
// LifeItem
// ========================================================================

TEST_F(LifeItemTest, AliveAtCreation) {
    const LifeItem item(100.f);

    ASSERT_FALSE(item.is_dead());
}

TEST_F(LifeItemTest, DeadWhenHpReachZero) {
    LifeItem item(10.f);

    item.receive_damages(10.f);

    ASSERT_TRUE(item.is_dead());
}

TEST_F(LifeItemTest, DeadWhenHpExceeded) {
    LifeItem item(5.f);

    item.receive_damages(100.f);

    ASSERT_TRUE(item.is_dead());
}

TEST_F(LifeItemTest, AliveAfterPartialDamages) {
    LifeItem item(100.f);

    item.receive_damages(50.f);

    ASSERT_FALSE(item.is_dead());
}

TEST_F(LifeItemTest, ReceiveDamagesReturnsActualDamage) {
    LifeItem item(10.f);

    const float received = item.receive_damages(7.f);

    ASSERT_FLOAT_EQ(received, 7.f);
}

TEST_F(LifeItemTest, ReceiveDamagesClampedToRemainingHp) {
    LifeItem item(5.f);

    const float received = item.receive_damages(20.f);

    ASSERT_FLOAT_EQ(received, 5.f);
}

TEST_F(LifeItemTest, ReceiveZeroDamages) {
    LifeItem item(100.f);

    const float received = item.receive_damages(0.f);

    ASSERT_FLOAT_EQ(received, 0.f);
    ASSERT_FALSE(item.is_dead());
}

TEST_F(LifeItemTest, MultipleDamagesAccumulate) {
    LifeItem item(10.f);

    item.receive_damages(3.f);
    item.receive_damages(3.f);
    item.receive_damages(3.f);

    ASSERT_FALSE(item.is_dead());

    item.receive_damages(1.f);

    ASSERT_TRUE(item.is_dead());
}

TEST_F(LifeItemTest, ZeroHpAtCreationIsDead) {
    const LifeItem item(0.f);

    ASSERT_TRUE(item.is_dead());
}

TEST_F(LifeItemTest, IsAlreadyDeadReturnsFalseFirstTime) {
    LifeItem item(1.f);

    item.receive_damages(1.f);
    ASSERT_TRUE(item.is_dead());

    ASSERT_FALSE(item.is_already_dead());
}

TEST_F(LifeItemTest, IsAlreadyDeadReturnsTrueSecondTime) {
    LifeItem item(1.f);

    item.receive_damages(1.f);

    item.is_already_dead();
    ASSERT_TRUE(item.is_already_dead());
}

TEST_F(LifeItemTest, IsAlreadyDeadReturnsFalseWhenAlive) {
    LifeItem item(100.f);

    ASSERT_FALSE(item.is_already_dead());
}

TEST_F(LifeItemTest, NoDamagesAfterDeath) {
    LifeItem item(5.f);

    item.receive_damages(5.f);
    const float received = item.receive_damages(10.f);

    ASSERT_FLOAT_EQ(received, 0.f);
}

// ========================================================================
// Item base
// ========================================================================

namespace {
    class DummyItem final : public Item {
    public:
        explicit DummyItem(const std::string &name) : Item(name) {}

        std::shared_ptr<Shape> get_shape() override { return nullptr; }
        glm::mat4 get_model_matrix() override { return glm::mat4(1.f); }
        glm::vec3 get_linear_velocity() override { return glm::vec3(0.f); }
        glm::vec3 get_angular_velocity() override { return glm::vec3(0.f); }

    protected:
        glm::vec3 _get_scale() override { return glm::vec3(1.f); }
    };
}// namespace

TEST_F(ItemBaseTest, GetNameReturnsConstructorName) {
    DummyItem item("my_item");

    ASSERT_EQ(item.get_name(), "my_item");
}

TEST_F(ItemBaseTest, NeedDestroyFalseByDefault) {
    DummyItem item("item");

    ASSERT_FALSE(item.need_destroy());
}

TEST_F(ItemBaseTest, DestroyThenNeedDestroy) {
    DummyItem item("item");

    item.destroy();

    ASSERT_TRUE(item.need_destroy());
}

TEST_F(ItemBaseTest, OnContactDoesNotCrash) {
    DummyItem item_a("a");
    DummyItem item_b("b");

    ASSERT_NO_THROW(item_a.on_contact(&item_b));
}

TEST_F(ItemBaseTest, TickDoesNotCrash) {
    DummyItem item("item");

    ASSERT_NO_THROW(item.tick());
}

// ========================================================================
// LifeItem — negative HP edge cases
// ========================================================================

TEST_F(LifeItemTest, NegativeHpAtCreationIsDead) {
    const LifeItem item(-5.f);

    ASSERT_TRUE(item.is_dead()) << "item with negative HP should be dead at creation";
}

TEST_F(LifeItemTest, NegativeHpReceiveDamagesReturnsZero) {
    LifeItem item(-5.f);

    const float received = item.receive_damages(10.f);

    ASSERT_FLOAT_EQ(received, 0.f)
        << "receiving damages on a negative-HP item should return 0 (already dead)";
}

TEST_F(LifeItemTest, NegativeHpIsAlreadyDeadFirstCallReturnsFalse) {
    LifeItem item(-5.f);

    // item is dead from creation, but is_already_dead should return false
    // the first time it's called (triggering the death event)
    ASSERT_TRUE(item.is_dead());
    ASSERT_FALSE(item.is_already_dead())
        << "first is_already_dead call on a dead item should return false (death event trigger)";
}

TEST_F(LifeItemTest, NegativeHpIsAlreadyDeadSecondCallReturnsTrue) {
    LifeItem item(-5.f);

    item.is_already_dead();
    ASSERT_TRUE(item.is_already_dead()) << "second is_already_dead call should return true";
}

// ========================================================================
// Item — double destroy
// ========================================================================

TEST_F(ItemBaseTest, DoubleDestroyDoesNotCrash) {
    DummyItem item("item");

    item.destroy();
    ASSERT_TRUE(item.need_destroy());

    ASSERT_NO_THROW(item.destroy());
    ASSERT_TRUE(item.need_destroy());
}
