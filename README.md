## tikitrace
Basic module for caveman-style PyTorch memory usage debugging

Are you trying to run PyTorch models without access to a supercomputer? Is your model using 28 yottabytes of memory when loading?

This module won't help you with either of those, but it might point fingers at what it thinks is eating memory.


### How to use
```py
import tikitrace

@tikitrace.trace()
def do_something_crazy():
    torch.load("blah blah blah")
```

crazy right? it gets even crazier:

```py
import tikitrace

with tikitrace.trace_ctx():
    torch.load("blah blah blah")
```

### It doesn't work!
you're probably right, file an issue and ill fix it

### All the measurements are inaccurate!
you're probably right, file an issue and ill fix it
