# Custom class -> (layer / activation function attributes) + forward pass
# Criterion -> Loss function
# Optimizers
# DataLoader class -> For iterating over batches of data; Training loader, Validation loader, Testing loader
 
criterion = nn.CrossEntropyLoss()
optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Custom Training loop
for epoch in range(100):
    loss = 0.0

    for i, data in enumerate(training_loader):
        # basic training loop
        inputs,  labels = data
        # set the gradient descent to zero initially
        optim.zero_grad()
        outputs = model(inputs)
        # calculate loss
        _loss = criterion(outputs, labels)
        # performe the backward pass
        loss.backward()
        # perform a single optimzation step
        optim.step()

        # increment the loss
        # loss.item() extracts a loss value as Python (float)
        loss =+ _loss.item()

        # Every 1000th mini-batch
        if i % 1000 == 999:
            print(f'Batch: {i+1}')
            
            # check for valition loss as well
            val_loss = 0.0
            # set training to fasle for validation
            model.train(False)
            
            for j, vdata in enumerate(validation_loader, 0):
                v_inputs, v_labels = vdata
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_labels)
                val_loss += v_loss.item()
            
            model.train(True)

            avg_train_loss = loss / 1000
            avg_val_loss = val_loss / 1000

            # Log the running loss avg / batch
            writer.add_scalers(
                    'Training v. Validation  loss',
                    {'Training': avg_train_loss,
                    'Validation': avg_val_loss
                    },
                    epoch * len(training_loader) + i)

            loss = 0.0
            

print('Finished Training')
writer.flush()
        
