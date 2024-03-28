import models, get_data, model_utils

def main():
    batch_size: int = 8
    num_epochs: int = 20
    width: int = 100
    height: int = 100

    model = models.LinearModel(width, height)
    #model = models.CNNModel(width,height)

    # this way of using train_loader and test_loader is completely fine
    train_loader, test_loader = get_data.get_data_loaders(amount=1000,width=width,height=height, batch_size=batch_size)
    print('\nDATA GENERATION COMPLETE\n')
    model = model_utils.train_model(model, train_loader, num_epochs=num_epochs)
    model_utils.eval_model(model, test_loader)
    
    X_test, y_test, y_pred = model_utils.get_preditions(model, test_loader)
    
    for i in range(len(X_test)):
        model_utils.plot_predictions(y_pred[i], y_test[i])
    model_utils.save_model(model)
    return
    

if __name__ == '__main__':
    main()