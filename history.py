import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('training history_trial_4.csv')

plt.style.use("ggplot")
fig0=plt.figure()
plt.plot(data["nb_epochs"], data["train_loss"], label="train")
plt.plot(data["nb_epochs"], data["test_loss"], label="test")
#plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.grid(False)


plt.tight_layout()
plt.show()
fig0.savefig('training loss history.png', dpi=300)