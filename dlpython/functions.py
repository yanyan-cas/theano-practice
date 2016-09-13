def similarity(image, images):
	similarities = []
	for img in images:
		distance = sqrt(sum(square(image - img)))
		sim = 1 / distance
		similarities.append(sim)
	return similarities

def ocr(img, images, labels):
	similarities = similarity(image, images)
	return labels[argmax(similarities)]
