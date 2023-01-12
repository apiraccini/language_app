from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

test_input = {'text': 'Ciao bella'}
test_expected = {'language': 'Italian'}

def test_main(input, expected):
    response = client.post("/predict", json=input)
    assert response.status_code == 200
    assert response.json() == expected

if __name__ == '__main__':
    test_main(test_input, test_expected)
    print('Test passed!')