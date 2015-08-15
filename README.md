What is it?
-----------

[markovify] is a great library for Markov chains in that it's very easy to
use. However, its approach is pretty memory-hungry. This code provides
a layer that can be wrapped aronud markovify in order to preprocess a model
and save it to disk.

If the caching is turned up to full, the advantages it provides are
somewhat marginal, although it seems to use _slightly_ less memory even when
everything is cached.

The primary use-case is where memory is scarce and millisecond-level latency
is not important. With conservative caching settings, it will generate a
sentence at least within a few seconds, while using dramatically less memory
than markovify. (This makes it easier to try applications where several
models are used at once.)

markovify: https://github.com/jsvine/markovify

Interface
---------

Use the function ```train_to_disk``` to train a model with markovify (or
any class that has a compatible interface) and save it to disk in a
format that this module can read.

Then use ```load_from_disk``` to read back a model from disk. The outer
layers of the class will be "retrained" (in the stock markovify code, this
is the part that guards against re-generating parts of the corpus verbatim),
but the core "chain" class will be replaced with an object that reads
from disk. The file handle will need to stay open as long as the model is
in use.

State of the code
-----------------

I'd call this a "pre-alpha". It was written to scratch a personal itch, and if
it's useful to anyone else that's great, but don't go in with any expectations.
