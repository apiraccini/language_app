from fastapi.testclient import TestClient
from main import app

# test client
client = TestClient(app)

# define test inputs and expected outputs
input1 = {'text': 'Ciao bella'}
expected1 = 'Italian'

input2 = {'text': 'Dirty little fucker'}
expected2 = 'Offensive'

def test_main():

    response = client.post('/predict_language', json=input1)
    assert response.status_code == 200
    assert response.json()['classification'] == expected1
    print('Language detection OK')

    response = client.post('/predict_offensive', json=input2)
    assert response.status_code == 200
    assert response.json()['classification'] == expected2
    print('Offensive english classification OK')

if __name__ == '__main__':
    test_main()
    print('Both tests passed!')