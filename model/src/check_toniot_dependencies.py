
import sys
print("Python:", sys.executable)

for name in ["numpy", "pandas", "xgboost", "sklearn"]:
    try:
        module = __import__(name)
        print(name, "OK", getattr(module, "__version__", ""))
    except Exception as exc:
        print(name, "FAILED:", repr(exc))
