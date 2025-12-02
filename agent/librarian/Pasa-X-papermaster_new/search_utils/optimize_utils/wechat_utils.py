import requests
import json
import os


def get_token(appID,appsecret):
    url_token = 'https://api.weixin.qq.com/cgi-bin/token?'
    res = requests.get(url=url_token,params={
             "grant_type": 'client_credential',
             'appid':appID,
             'secret':appsecret,
             }).json()
    # print(res)
    token = res.get('access_token')
    # print(res)
    return token

#读取配置文件
def rdconfig(filename):
    if (os.path.exists(filename)):
        with open(filename,"r") as f:
            config = json.load(f)
            # print("读取配置文件完成...")
            return config
    else:
        print('初始化配置文件')
        with open(filename, "w") as f:
            appID = input('appID: ')
            appsecret = input('appsecret: ')
            userid = input('userid: ')
            template_id = input('template_id: ')
            token = get_token(appID, appsecret)
            res = {"appID":appID,"appsecret":appsecret,"token":token,"userid":userid,"template_id":template_id}
            res1 = json.dumps(res)
            f.write(str(res1))
            f.close()
            return res

#写入配置文件
def wrconfig(new_dict,filename):
    with open(filename, "w") as f:
        json.dump(new_dict, f)
    # print("写入配置文件完成...")

def sendtext(token,userID,text):
    url_msg = 'https://api.weixin.qq.com/cgi-bin/message/custom/send?'
    body = {
        "touser": userID,  # 这里必须是关注公众号测试账号后的用户id
        "msgtype": "text",
        "text": {
            "content": text
        }
    }
    res = requests.post(url=url_msg, params={
        'access_token': token  # 这里是我们上面获取到的token
    }, data=json.dumps(body, ensure_ascii=False).encode('utf-8'))
    return res

def sendmb(token,template_id,userID,text,color):
    url_msg = 'https://api.weixin.qq.com/cgi-bin/message/template/send?'
    body = {
        "touser": userID,  # 这里必须是关注公众号测试账号后的用户id
        "template_id": template_id,
        "url": "http://weixin.qq.com/download",
        "topcolor": color,
        "data": {
            "text":{
                "value": text,
                "color": color
            }
        }
    }
    res = requests.post(url=url_msg, params={
        'access_token': token  # 这里是我们上面获取到的token
    }, data=json.dumps(body, ensure_ascii=False).encode('utf-8'))
    return res

def send(text):
    config = rdconfig('pasa1/utils/wechat_token.json')
    userID = config['userid']
    # userID = 'o-ZE8vowLRoN7r7nSgrtQNlq3Qwk'
    template_id =config['template_id']
    res = sendtext(config['token'],userID,text)
    if(res.json()['errcode']==42001): # token两小时过期后重新获取
        config['token']=get_token(config['appID'],config['appsecret'])
        wrconfig(config,'pasa1/utils/wechat_token.json')
        res = sendtext(config['token'],userID,text)
    if(res.json()['errcode']==45047 or res.json()['errcode']==45015): # 客服消息如果长时间不回复将不能发，这边先换成模板消息
        res = sendmb(config['token'],template_id,userID,text+'（请及时回复以免消息发送失败）','#FF0000')
    return res.json()

# print(send('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
# 发送模板消息
# sendmb(token,template_id,userID,'程序错误，请及时回复','#FF0000')
