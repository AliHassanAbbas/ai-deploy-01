"""The smallest possible containerized program."""
import platform
import sys

print("Hello from inside a container!")
print(f"Python : {sys.version.split()[0]}")
print(f"OS     : {platform.system()} {platform.release()}")