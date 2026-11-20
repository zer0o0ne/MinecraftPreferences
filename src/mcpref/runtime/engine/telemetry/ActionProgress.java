package mcpref.runtime;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.lang.reflect.Field;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.PlayerController;
import net.minecraft.entity.LivingEntity;
import net.minecraft.item.CrossbowItem;
import net.minecraft.item.ItemStack;
import net.minecraft.item.UseAction;
import net.minecraft.util.math.BlockPos;

/** Read-only progress observations for specs/runtime-stagnation.md. */
public final class ActionProgress {
    private static final Field TARGET = field("currentBlock");
    private static final Field DAMAGE = field("curBlockDamageMP");

    private static Field field(String name) {
        try {
            Field result = PlayerController.class.getDeclaredField(name);
            result.setAccessible(true);
            return result;
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException("incompatible mining telemetry", error);
        }
    }

    public static void append(JsonObject observation) {
        Minecraft game = Minecraft.getInstance();
        JsonObject progress = new JsonObject();
        progress.addProperty("version", 1);
        progress.add("mining", mining(game.playerController));
        progress.add("item_use", itemUse(game.player));
        observation.add("action_progress", progress);
    }

    private static JsonObject mining(PlayerController controller) {
        try {
            boolean active = controller.getIsHittingBlock();
            BlockPos target = active ? (BlockPos) TARGET.get(controller) : BlockPos.ZERO;
            JsonArray coordinates = new JsonArray();
            coordinates.add(target.getX());
            coordinates.add(target.getY());
            coordinates.add(target.getZ());
            JsonObject result = new JsonObject();
            result.addProperty("active", active);
            result.add("target", coordinates);
            result.addProperty("progress", active ? DAMAGE.getFloat(controller) : 0.0f);
            return result;
        } catch (IllegalAccessException error) {
            throw new IllegalStateException("cannot read mining telemetry", error);
        }
    }

    private static JsonObject itemUse(LivingEntity player) {
        boolean active = player.isHandActive();
        ItemStack item = player.getActiveItemStack();
        UseAction action = active ? item.getUseAction() : UseAction.NONE;
        int duration = active ? item.getUseDuration() : 0;
        JsonObject result = new JsonObject();
        result.addProperty("active", active);
        result.addProperty("action", action.name());
        result.addProperty("duration", duration);
        result.addProperty("charge_duration", chargeDuration(item, action, duration));
        result.addProperty("elapsed", active ? player.getItemInUseMaxCount() : 0);
        result.addProperty("remaining", active ? player.getItemInUseCount() : 0);
        return result;
    }

    private static int chargeDuration(ItemStack item, UseAction action, int duration) {
        // Pinned BowItem.getArrowVelocity and TridentItem.onPlayerStoppedUsing endpoints.
        // CrossbowItem supplies the enchantment-dependent endpoint directly.
        if (action == UseAction.EAT || action == UseAction.DRINK) return duration;
        if (action == UseAction.BOW) return 20;
        if (action == UseAction.SPEAR) return 10;
        if (action == UseAction.CROSSBOW) return CrossbowItem.getChargeTime(item);
        return 0;
    }
}
