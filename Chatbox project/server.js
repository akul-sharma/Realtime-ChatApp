const http = require('http')
const express = require('express')
const app = express()
const socketio = require('socket.io')

const server = http.createServer(app)
const io = socketio(server)

app.use(express.static(__dirname + '/public'))

users = {
   akul : 'abc123'
}

let socketMap = {}

function login(s,u) {
    s.join(u)
    s.emit('logged_in')
    socketMap[s.id] = u
    console.log(socketMap)
}

io.on('connection' , (socket) => {
    console.log('Connected with Socket id :: ', socket.id)

    socket.on('login' , (data) => {

        if(users[data.username]){
            if(users[data.username] == data.password) {
                login(socket,data.username)
            }
            else {
                socket.emit('login_failed')
            }
        }
        else{
            users[data.username] = data.password
           login(socket,data.username)
        }
        console.log(users)
    })

    socket.on('msg_send' , (data) => {
        data.from = socketMap[socket.id]
        if(data.to) {
            io.to(data.to).emit('msg_rcvd' , data)
        }
        else{
            socket.broadcast.emit('msg_rcvd' , data)
        }
    })
})


server.listen(3344 , ()=> { 
    console.log('Server Started On http://localhost:3344')
})