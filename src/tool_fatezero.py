import task


def main(arg: tuple[str, task.FateZero]):
    id_, a = arg

    print("FateZero: \"" + a.prompt.replace("\n", "\\n") + "\"")
