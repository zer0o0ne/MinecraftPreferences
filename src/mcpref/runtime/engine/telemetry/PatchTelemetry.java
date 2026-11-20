package mcpref.runtime;

import java.nio.file.Files;
import java.nio.file.Paths;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;

/** Add one observation hook without changing gameplay or control flow. */
public final class PatchTelemetry {
    public static void main(String[] arguments) throws Exception {
        ClassReader reader = new ClassReader(Files.readAllBytes(Paths.get(arguments[0])));
        ClassNode node = new ClassNode(Opcodes.ASM7);
        reader.accept(node, 0);
        int hooks = 0;
        for (MethodNode method : node.methods) {
            if (method.name.equals("getInfo") &&
                method.desc.equals("(Ljava/lang/String;)Ljava/lang/String;")) {
                hooks += patch(method);
            }
        }
        if (hooks != 1) {
            throw new IllegalStateException("expected one observation hook, got " + hooks);
        }
        ClassWriter writer = new ClassWriter(reader, ClassWriter.COMPUTE_MAXS);
        node.accept(writer);
        Files.write(Paths.get(arguments[1]), writer.toByteArray());
    }

    private static int patch(MethodNode method) {
        int hooks = 0;
        for (AbstractInsnNode instruction : method.instructions.toArray()) {
            if (instruction.getOpcode() != Opcodes.ARETURN) continue;
            AbstractInsnNode previous = instruction.getPrevious();
            while (previous != null && previous.getOpcode() < 0) {
                previous = previous.getPrevious();
            }
            if (!(previous instanceof MethodInsnNode)) continue;
            MethodInsnNode call = (MethodInsnNode) previous;
            if (call.getOpcode() != Opcodes.INVOKEVIRTUAL ||
                !call.owner.equals("com/google/gson/JsonObject") ||
                !call.name.equals("toString") ||
                !call.desc.equals("()Ljava/lang/String;")) continue;
            InsnList hook = new InsnList();
            hook.add(new InsnNode(Opcodes.DUP));
            hook.add(new MethodInsnNode(Opcodes.INVOKESTATIC,
                "mcpref/runtime/ActionProgress", "append",
                "(Lcom/google/gson/JsonObject;)V", false));
            method.instructions.insertBefore(call, hook);
            hooks++;
        }
        return hooks;
    }
}
