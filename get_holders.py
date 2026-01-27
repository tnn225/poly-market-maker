import time

from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.utils.telegram import Telegram
from poly_market_maker.my_token import MyToken

setup_logging()

clob_api = ClobApi()
telegram = Telegram()

SIDES = {
    MyToken.A: 'Up',
    MyToken.B: 'Down'
}
SYMBOLS = ['btc', 'eth', 'sol', 'xrp']
MIN_HOLDER_SIZE = 10000

once = {} 

def is_ok(holders: list) -> bool:
    for holder in holders:
        if float(holder['amount']) > MIN_HOLDER_SIZE:
            return True
    return False

def is_good(holder: dict) -> bool:
    return float(holder['amount']) > MIN_HOLDER_SIZE

def get_size(holders: list) -> float:
    size = 0
    for holder in holders:
        size += float(holder['amount'])
    return size

def print_holders(interval: int, symbol: str):
    print(f"Printing holders for interval: {interval}")

    market = clob_api.get_market(interval, symbol)
    rows = clob_api.get_holders(market)
    for my_token, side in SIDES.items():
        token_id = market.token_id(my_token)
        for row in rows:
            # print(f"row: {row}")
            if int(row['token']) != token_id:
                continue
            holders = row['holders']
            if not is_ok(holders):
                continue
            slug = f"{symbol}-updown-15m-{interval}"

            message = ""
            for holder in holders:
                # print(f"holder: {holder}")
                if not is_good(holder):
                    continue
                key = f"{slug}-{side}-{holder['proxyWallet']}"
                if once.get(key):
                    continue
                once[key] = True
                text = f"""<a href="https://polymarket.com/profile/{holder['proxyWallet']}">{holder['name']}</a> {holder['amount']:.2f} {side} {slug}\n"""
                message += text
            if len(message) > 0:
                telegram.send_message(message, disable_web_page_preview=True)

def run():
    interval = int(time.time() // 900 * 900)
    for symbol in SYMBOLS:
        print_holders(interval, symbol)

def main():
    run()
    while True:
        time.sleep(1)
        now = int(time.time()) + 10
        if (now % 60) == 0:
            try:
                run()
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)
        
if __name__ == "__main__":
    main()

"""
row: {
   "token":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
   "holders":[
      {
         "proxyWallet":"0xe2a432ec3e4b8be701359bff9e72f470c19c0fe6",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Mixed-Pomelo",
         "amount":208.136265,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"071206",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xfec0df6bfe91082bd956ebeb79c8e324d4781e6c",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Official-Gasp",
         "amount":90.245502,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"0x3abde56a23c08ac1ee9",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xa9650fe4301f45e7f090ada7252f9c1268183565",
         "bio":"happy betting",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Woeful-Analyst",
         "amount":90,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"TortoiseMaker",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xe594336603f4fb5d3ba4125a67021ab3b4347052",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Weak-Finer",
         "amount":63.639732,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"0xE594336603F4fB5d3ba4125a67021ab3B4347052-1769022918519",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x6e3d10c0420d2a91f2c852e58a37e30255e3c636",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Aware-Mesenchyme",
         "amount":49.4701,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"lancewu1910",
         "profileImage":"https://polymarket-upload.s3.us-east-2.amazonaws.com/profile-image-4205289-2ebbb061-a206-4515-a9b0-0bc8d413cea4.jpg",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x4661272baa178f5cc361283d8a35071eaa4dd3d9",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Ugly-Counselor",
         "amount":37.5322,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"0x2b1f",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xa8b158111a260db32518240b0f481ad6141451c1",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Strange-Moment",
         "amount":34.140557,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"errmmm",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xa1fc36bf45d3e56a00367dd087f0b42d4374b935",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Limp-Oar",
         "amount":31.226414,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"0xA1FC36bF45d3E56A00367dd087F0B42D4374B935-1763994215877",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x6e1baa831f82cfa92ba1c9dfcbde5105ca728ad8",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"",
         "amount":27.190157,
         "displayUsernamePublic":false,
         "outcomeIndex":1,
         "name":"",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x2eca63a5c08161214914f445df0373a78ce8ec32",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Delicious-Tailbud",
         "amount":25.911875,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"AchelDrinker",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x6748a5aa767cf0f51e629c680916ffa31e3ac5b4",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Sad-Spelling",
         "amount":22.7563,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"0x2abf494b2780cf5335e1b",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x86a29f88fcc23ea2b0e01b4e186b043e26c873a8",
         "bio":"Is it too late now to say gnaw-ry",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Quick-Sushi",
         "amount":20.939022,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"justinbeaver",
         "profileImage":"https://polymarket-upload.s3.us-east-2.amazonaws.com/profile-image-3679094-2d5bc64d-36ca-4410-ad7e-4d253122abd4.webp",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x0d812103a029cf8800f2e3c3bfbcbebdbbd07edf",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Far-Off-Recording",
         "amount":20,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"ephialtes",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x3f348100519813dd4ba2a9aff346de5a0f42ed6a",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Flimsy-Nightgown",
         "amount":19.8754,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"jackccstream",
         "profileImage":"https://polymarket-upload.s3.us-east-2.amazonaws.com/profile-image-3969103-c259caae-bb3e-48fe-92df-f40fe9e6eb79.jpg",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xe7babf65fe5998a01875745fad78375d6a435f37",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Pertinent-Processing",
         "amount":13.4147,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"Paracelsus1929",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xe55b90febea370d4611a4b0a9ff201183268b24e",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Agitated-Hovercraft",
         "amount":12.65181,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"..47plum",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0xf6963d4cdbb6f26d753bda303e9513132afb1b7d",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Junior-Octavo",
         "amount":11.743861,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"rename",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x9094c89f2567315d37e9fc78ac71722fbe2ddcab",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Delirious-Prejudice",
         "amount":11.382674,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"Troy168",
         "profileImage":"https://polymarket-upload.s3.us-east-2.amazonaws.com/profile-image-4549984-4d7070b0-5833-42ed-96d5-721083d947c1.jpg",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x2d9b3c3af82dc64c7d52daedd090c8378176085e",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Oddball-Minimalism",
         "amount":10,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"xiaotiantian2",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      },
      {
         "proxyWallet":"0x80015385398e8bee141186404d7c8e30a5959718",
         "bio":"",
         "asset":"95640479183913974198647115801293521155817668491623120867168842645317553583105",
         "pseudonym":"Bronze-Strategy",
         "amount":10,
         "displayUsernamePublic":true,
         "outcomeIndex":1,
         "name":"beefstew",
         "profileImage":"",
         "profileImageOptimized":"",
         "verified":false
      }
   ]
}
"""