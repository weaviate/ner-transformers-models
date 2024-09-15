import unittest
import requests
import time

class SmokeTest(unittest.TestCase):
    def _waitForStartup(self):
        url = 'http://localhost:8000/.well-known/ready'

        for i in range(0, 100):
            try:
                res = requests.get(url)
                if res.status_code == 204:
                    return
                else:
                    raise Exception(
                            "status code is {}".format(res.status_code))
            except Exception as e:
                print("Attempt {}: {}".format(i, e))
                time.sleep(1)

        raise Exception("did not start up")

    def testNer(self):
        self._waitForStartup()
        url = 'http://localhost:8000/ner/'

        req_body = {'text': 'John works in Berlin'}
        res = requests.post(url, json=req_body)
        resBody = res.json()
        
        expected_result = {'text': 'John works in Berlin', 'tokens': [{'word': 'John', 'entity': 'I-PER', 'certainty': 0.9974473714828491, 'startPosition': 0, 'endPosition': 4}, {'word': 'Berlin', 'entity': 'I-LOC', 'certainty': 0.9994735717773438, 'startPosition': 14, 'endPosition': 20}]}
        
        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertEqual(len(resBody['tokens']), len(expected_result['tokens']))

        req_body = {'text': 'Hello how are you doing'}
        res = requests.post(url, json=req_body)
        resBody = res.json()

        expected_result = {'text': 'Hello how are you doing', 'tokens': []}
        
        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertCountEqual(resBody['tokens'], expected_result['tokens'])

        req_body = {'text': ''}
        res = requests.post(url, json=req_body)
        resBody = res.json()

        expected_result = {'text': '', 'tokens': None}

        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertTrue(resBody['tokens'] is None or resBody['tokens'] == [])


if __name__ == "__main__":
    unittest.main()
