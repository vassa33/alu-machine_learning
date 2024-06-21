#!/usr/bin/env python3
"""
Loop to handle user input
"""

while True:
    print("Q: ", end="")
    question = input()

    if (("exit" in question.lower().strip()
         or "quit" in question.lower().strip()
         or "bye" in question.lower().strip())):
        print("A: Goodbye")
        exit()

    print("A: ")
