import os


if __name__ == '__main__':

    data_path = os.path.join(os.getcwd(), 'data', 'BMSWebView1', 'BMSWebView1.txt')
    with open(data_path, 'r') as f:
        dataset = f.read()
    dataset = dataset.split('-2\n')
    print(dataset)
