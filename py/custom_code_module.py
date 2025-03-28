from pathlib import Path

# GLOBALS

ROOT = Path(__file__).resolve().parent.parent
ROOT_COMFY = ROOT.parent.parent
ROOT_FONTS = ROOT / "fonts"

# SUPPORT CLASSES


class AlwaysEqualProxy(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

# SUPPORT FUNCTIONS


def parse_dynamic(data: dict, key: str) -> list:
    vals = []
    count = 1
    while data.get((who := f"{key}_{count}"), None) is not None:
        vals.append(who)
        count += 1
    if len(vals) == 0:
        vals.append([])
    return vals


class CodeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "code_input": ("STRING", {"default": "outputs[0] = 'hello, world!'", "multiline": True, "dynamicPrompts": False}),
                "file": ("STRING", {"default": "./res/hello.py", "multiline": False, "dynamicPrompts": False}),
                "use_file": ("BOOLEAN", {"default": False}),
                "run_always": ("BOOLEAN", {"default": False})
            }}

    CATEGORY = "model_patches/bending"
    EXPERIMENTAL = True

    RETURN_TYPES = ("BENDING_MODULE", "STRING", )
    RETURN_NAMES = ("BENDING_MODULE", "DEBUG", )
    DESCRIPTION = """
    CodeNode is designed to execute custom user-provided Python code. The code can be directly entered as a string input or loaded from a specified file. This class processes dynamic inputs and provides four generic output slots. The execution environment includes predefined 'inputs' and 'outputs' dictionaries to facilitate interaction with the code. Proper error handling is included to ensure informative feedback in case of execution failures. This node is ideal for users needing to integrate custom logic or algorithms into their workflows.
    """
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(self, code_input, file, use_file, run_always, **kwargs):
        if run_always:
            return float('nan')
        hash = '$$' + str(kwargs) + '$$' + self.get_exec_string(self,
                                                                code_input, file, use_file) + '$$'
        # print(hash)
        return hash

    def execute(self, code_input, file, use_file, run_always, **kwargs):
        outputs = {i: None for i in range(4)}
        inputs = kwargs.copy()
        inputs.update({i: v for i, v in enumerate(kwargs.values())})
        code = self.get_exec_string(code_input, file, use_file)

        try:
            exec(code, {"inputs": inputs, "outputs": outputs})
        except Exception as e:
            raise RuntimeError(f"Error executing user code: {e}")

        return tuple(outputs[i] for i in range(4))

    def get_exec_string(self, code_input, file, use_file):
        if use_file:
            # load the referenced file
            code_input = ""
            if not (fname := Path(ROOT / file)).is_file():
                # print(fname)
                if not (fname := Path(file)).is_file():
                    # print(fname)
                    fname = None
            if fname is not None:
                try:
                    with open(str(fname), 'r') as f:
                        code_input = f.read()
                except Exception as e:
                    raise RuntimeError(
                        f"[FL_CodeNode] error loading code file: {e}")
            # print(code_input)
        return code_input
