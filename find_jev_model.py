"""Find valid TypeSafe model names."""
import sys, os
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(".env")

jev_api_key = os.environ.get("JEV_API_KEY")
from typesafe_sdk import TypeSafeClient, TypeSafeError, Choice

# Try different model names
for model_name in ["jev-1", "jev-2", "default", "latest", "gpt-4o-mini", "typesafe-1", "jev-mini"]:
    try:
        with TypeSafeClient(api_key=jev_api_key, model=model_name) as client:
            response = client.system_one(
                state={"test": "hello"},
                questions={"q": Choice(instructions="Is this a test?", criteria={"yes": "Yes", "no": "No"})},
            )
        print(f"Model {model_name!r}: SUCCESS")
        break
    except TypeSafeError as e:
        print(f"Model {model_name!r}: {e}")
    except Exception as e:
        print(f"Model {model_name!r}: OTHER ERROR: {type(e).__name__}: {e}")
