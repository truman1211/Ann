import torch




class problem1_loss(torch.nn.Module):
    def __init__(self):
        super(problem1_loss, self).__init__()
    def forward(self,ann, right_hand_function,coefficient_function,
                ic, training_data):
             x = training_data
             """
             there is an other way to get dtrial_solution_x
             """
             # def trial_solution(x):
             #
             #    trial_solution = ic + x*ann(x)
             #
             #    return trial_solution
             #
             # eps = 1e-5
             # dtrial_solution_x = (trial_solution(x+eps)-trial_solution(x))/eps

             ann(x).backward(torch.ones_like(x))


             dtrial_solution_x = ann(x)+x*x.grad

             x.grad.data.zero_()

             error_list = (dtrial_solution_x - (right_hand_function(x)-coefficient_function(x)*trial_solution(x)))**2

             loss = torch.sum(error_list)


             return loss

def coefficient_function(x):
    f = x + (1+3*(x**2))/(1+x+x**3)

    return f
def right_hand_function(x):

    f = x**3 +2*x + (x**2)* ((1+3*(x**2)) /(1+x+x**3))

    return f







