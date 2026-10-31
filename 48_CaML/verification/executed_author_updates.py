meta_optimizer.zero_grad() # reset meta optimizer

 # anneal meta-lr
frac_done = float(iteration) / n_iterations
new_lr = frac_done * meta_lr_final + (1 - frac_done) * meta_lr
for pg in meta_optimizer.param_groups:
    pg['lr'] = new_lr

# zero-grad the parameters
for p in model.parameters(): # following l2l example
    p.grad = torch.zeros_like(p.data)

learner = copy.deepcopy(model) # model to adapt
# init new optimizer for task
inner_optimizer = get_optimizer(learner, args.learning_rate, args.weight_decay)
inner_optimizer.load_state_dict(inner_optimizer_state)

# Adapt model to current task:
inner_val_loss = train_adapter(
    train_loader,
    val_loader,
    learner,
    inner_optimizer,
)

# Get new optimizer state:
inner_optimizer_state = inner_optimizer.state_dict()
# Update gradient of outer model for SGD: Add Phi tilde component
for p, l in zip(model.parameters(), learner.parameters()):
    p.grad.data.add_(-1.0, l.data )
meta_train_loss += inner_val_loss.item()

for p in model.parameters():
    p.grad.data.mul_(1.0 / meta_batch_size).add_(p.data)
meta_optimizer.step()
meta_optimizer.zero_grad()
