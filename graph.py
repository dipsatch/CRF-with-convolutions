import matplotlib.pyplot as plt

def get_graph(accuracy, c, title):
	plt.figure(figsize=(len(c),len(accuracy)), dpi=80)
	plt.plot(c, accuracy,  color="red") 
	plt.xticks(c)
	plt.xlabel("c")
	plt.ylabel("Accuracy %")
	plt.title(title)
	plt.show()
