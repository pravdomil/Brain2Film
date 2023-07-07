import task


def main(a: task.BarkVoice2Voice):
    print("Bark voice2voice: \"" + a.prompt.replace("\n", "\\n") + "\"")
