from typing import Any, Optional

from .log import log


class Patch:
    def __init__(
        self,
        base: Any,
        key: str,
        patched: Any,
        exit_hook: Any = None,
        enable: bool = True,
        context: Optional[str] = None,
        **kwargs
    ) -> None:
        self.base = base
        self.key = key
        self.enable = enable
        self.unpatched = getattr(base, key)
        self.patched = lambda *args, **p_kwargs: patched(self, *args, **p_kwargs)
        self.exit_hook = exit_hook
        self.context = context
        self.kwargs = kwargs

    def log(self, *args):
        log(*args, context=self.context or self.target_name)

    @property
    def target_name(self) -> str:
        try:
            return f"{self.base.__qualname__}.{self.key}"
        except AttributeError:
            return f"{self.base.__name__}.{self.key}"

    def __enter__(self) -> None:
        if self.enable:
            setattr(self.base, self.key, self.patched)

    def __exit__(self, *args) -> None:
        setattr(self.base, self.key, self.unpatched)
        if self.exit_hook:
            self.exit_hook(self)


class PatchList(list):
    def __init__(self, listlike):
        for x in listlike:
            if not isinstance(x, Patch):
                raise ValueError(
                    f"Tried to instantiate PatchList with value of type '{type(x).__name__}'"
                )
        return super().__init__(listlike)

    def __enter__(self) -> None:
        for patch in self:
            patch.__enter__()

    def __exit__(self, *args) -> None:
        for patch in self:
            patch.__exit__()

    def __setitem__(self, key, value) -> None:
        if not isinstance(value, Patch):
            raise ValueError(
                f"Tried to add value of type '{type(value).__name__}' to PatchList"
            )
        return super().__setitem__(key, value)
