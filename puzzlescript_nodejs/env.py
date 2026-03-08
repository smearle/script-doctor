from puzzlejax.backends.nodejs import NodeJSPuzzleScriptBackend

backend = NodeJSPuzzleScriptBackend()
engine = backend.engine
solver = backend.solver


# TODO: Wrap standalone nodejs engine in a gym environment
