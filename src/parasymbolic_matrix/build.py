from swimport import FileSource, Swim


def write_swim():
    src = FileSource("parasymbolic.h")
    swim = Swim("parasymbolic")



    swim.write("parasymbolic.i")

