def RSS(response):

    # Obtain the main response
    mean_response = response.mean()

    # Compute the RSS
    return sum((response - mean_response)**2), mean_response

def RMSE(y, y_hat):
    return sum((y - y_hat)**2)**(1/2)