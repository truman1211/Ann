class coe_right_function():

    def coefficient_function(x):
        f = x + (1 + 3 * (x ** 2)) / (1 + x + x ** 3)

        return f

    def right_hand_function(x):
        f = x ** 3 + 2 * x + (x ** 2) * ((1 + 3 * (x ** 2)) / (1 + x + x ** 3))

        return f