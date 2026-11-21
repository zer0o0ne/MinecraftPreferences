"""Autograd and measured updates independently check serialized gradient diagnostics."""

import torch

from mcpref.training.pretrain.continuous.diagnostics.gradients import Gradients


class Capture:
    def start(self, context):
        self.flow = Gradients(context)
        self.records, self.vectors = {}, {}
        self.parameters = {
            f"{name}.{key}": parameter
            for name, module in context.modules.items()
            for key, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    def before_backward(self, context, components, events):
        vectors = self.vectors.setdefault(context.step, {})
        for component, loss in components.items():
            values = torch.autograd.grad(
                loss,
                list(self.parameters.values()),
                retain_graph=True,
                allow_unused=True,
                materialize_grads=False,
            )
            target = vectors.setdefault(component, {})
            for (name, parameter), gradient in zip(self.parameters.items(), values):
                value = torch.zeros_like(parameter) if gradient is None else gradient
                target[name] = target.get(name, 0) + value.detach().double()
        self.flow.collect(context, components)

    def before_update(self, context, result):
        vectors = self.vectors.get(context.step, {})
        self.before = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        for name, parameter in self.parameters.items():
            actual = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad
            )
            expected = sum(
                (component[name] for component in vectors.values()),
                torch.zeros_like(actual).double(),
            )
            torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-6)

    def after_update(self, context, result):
        self.flow.finish_step(context, result)
        if self.flow.pending:
            record = self.flow.pending[-1]
            if record["step"] == context.step:
                self.records[context.step] = record
                self.check(context, record)

    def check(self, context, record):
        vectors = self.vectors.get(context.step, {})
        phi_names = [
            name for name in self.parameters if name.startswith(("phi.", "phi_loss."))
        ]
        p_names = [name for name in self.parameters if name.startswith("P.")]

        def vector(names, component):
            return torch.cat(
                [
                    vectors.get(component, {})
                    .get(name, torch.zeros_like(self.parameters[name]))
                    .double()
                    .flatten()
                    for name in names
                ]
            )

        def verify(bundle, names):
            values = {key: vector(names, key) for key in ("phi", "pi", "P")}
            for key, value in values.items():
                assert_close(bundle["norms"][key], value.norm().item())
            others = {
                "phi": values["phi"],
                "pi": values["pi"],
                "other": values["phi"] + values["pi"],
                "joint": sum(values.values()),
            }
            for key, value in others.items():
                assert_close(
                    bundle["dot"][f"P_{key}"], torch.dot(values["P"], value).item()
                )

        verify(record["phi"]["total"], phi_names)
        assert_close(record["P"]["total"]["norm"], vector(p_names, "P").norm().item())
        for group, parameters in context.groups.items():
            names = [context.groups.parameter_names[id(p)] for p in parameters]
            phi = [name for name in names if name in phi_names]
            pref = [name for name in names if name in p_names]
            if phi:
                verify(record["phi"]["groups"][group], phi)
            for kind, selected in (("phi", phi), ("P", pref)):
                if not selected:
                    continue
                delta = torch.cat(
                    [
                        (self.parameters[name].detach() - self.before[name])
                        .double()
                        .flatten()
                        for name in selected
                    ]
                )
                expected = torch.dot(vector(selected, "P"), delta).item()
                assert_close(
                    record["update_alignment"][kind][group]["update_dot"], expected
                )

    def checkpoint(self, context, staged, results):
        self.flow.checkpoint(staged)


def assert_close(actual, expected):
    torch.testing.assert_close(
        torch.tensor(actual, dtype=torch.float64),
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
    )
