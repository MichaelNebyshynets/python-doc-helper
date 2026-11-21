// Запуск:  npm test   (простая проверка без фреймворков)
import assert from "node:assert";
import { createParser } from "./sse.js";

const got = [];
const feed = createParser((e, d) => got.push([e, d]));
const msg = 'event: token\ndata: {"t":"При\\nвет"}\n\nevent: done\ndata: {"cited":[1]}\n\n';
for (let i = 0; i < msg.length; i += 7) feed(msg.slice(i, i + 7)); // режем поток на кусочки по 7 символов
assert.deepStrictEqual(got, [["token", { t: "При\nвет" }], ["done", { cited: [1] }]]);

const got2 = [];
createParser((e, d) => got2.push([e, d]))("event: x\r\ndata: {\"a\":1}\r\n\r\n");
assert.deepStrictEqual(got2, [["x", { a: 1 }]]);
console.log("sse: ok");
