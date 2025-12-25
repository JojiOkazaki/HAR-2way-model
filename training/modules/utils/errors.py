import inspect

def error(message):
    frame = inspect.currentframe().f_back

    func_name = frame.f_code.co_name

    # クラスメソッドかどうか判定
    if "self" in frame.f_locals:
        class_name = frame.f_locals["self"].__class__.__name__
        location = f"{class_name}.{func_name}"
    else:
        location = func_name

    raise Exception(f"[{location}] {message}")
