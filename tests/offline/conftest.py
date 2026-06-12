"""Test-only PyICU substitute for environments without the optional native wheel."""

import sys
import types

try:
    import icu  # noqa: F401
except ImportError:

    class Script:
        @staticmethod
        def getScript(cp):
            return cp

        @staticmethod
        def getName(cp):
            ranges = (
                (0x0900, 0x097F, "Devanagari"),
                (0x0980, 0x09FF, "Bengali"),
                (0x0A00, 0x0A7F, "Gurmukhi"),
                (0x0A80, 0x0AFF, "Gujarati"),
                (0x0B80, 0x0BFF, "Tamil"),
                (0x0C00, 0x0C7F, "Telugu"),
                (0x0C80, 0x0CFF, "Kannada"),
                (0x0D00, 0x0D7F, "Malayalam"),
                (0x0600, 0x06FF, "Arabic"),
            )
            for start, end, name in ranges:
                if start <= cp <= end:
                    return name
            if 0x0041 <= cp <= 0x007A:
                return "Latin"
            return "Common"

    sys.modules["icu"] = types.SimpleNamespace(Script=Script)
