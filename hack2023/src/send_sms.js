// Download the helper library from https://www.twilio.com/docs/node/install
// Find your Account SID and Auth Token at twilio.com/console
// and set the environment variables. See http://twil.io/secure
const accountSid = 'AC4145f33db19c8adff71fab19c45859e1';
const authToken = '3c9810f77e383ed12cb1d14a96898c34';
const client = require('twilio')(accountSid, authToken);

client.messages
  .create({
     body: 'Hello',
     from: '+14694614640',
     to: '+14695697885'
   })
  .then(message => console.log(message.sid));