# `name` is the name of the package as used for `pip install package`
name = "supermage"
# `path` is the name of the package for `import package`
path = name.lower().replace("-", "_").replace(" ", "_")
# Your version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
version = "0.1.5"
author = "Michael James Yantovski Barth"
author_email = "mjb299@pitt.edu"
description = "SMBH masses from ALMA observations of gas kinematics"  # One-liner
url = ""  # your project homepage
license = "Unlicense"  # See https://choosealicense.com
