'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('web/app.js','utf8').split('// BEGIN PLACEMENT GEOMETRY')[1].split('// END PLACEMENT GEOMETRY')[0].split('\n').slice(1).join('\n');
const context={};vm.createContext(context);vm.runInContext(source,context);
const cfg={room_m:[6,6,3],clearance_m:.08,tx_m:[[.1,1,.45],[.1,5,.65],[5.9,1,.65],[5.9,5,.45]],rx_m:[[.1,2.6,.45],[.1,4.2,.65],[1.8,5.9,.45],[4.2,5.9,.65],[5.9,3.4,.45],[5.9,1.8,.65],[4.2,.1,.45],[1.8,.1,.65]]};
cfg.fixed_objects=[{name:'cabinet',x:3,y:5.72,size:[1.2,.38,.9]}];
const sizes=[[.6,.5,.8],[1.1,.8,1.1],[1.6,1,1.4]],obj=(kind,x,y)=>({kind,x,y,size:sizes[kind]});
const scenarios=[{name:'radio overlap',objects:[obj(0,3,3),obj(2,5,5)],i:1,x:5.15,y:3.4},{name:'radio exact clearance boundary',objects:[obj(0,3,3),obj(2,5,5)],i:1,x:5.02,y:3.4},{name:'peer overlap with same-zone solution',objects:[obj(0,3,3),obj(2,5,5)],i:1,x:3,y:3},{name:'outside room',objects:[obj(0,3,3),obj(2,5,5)],i:1,x:7,y:-1},{name:'small radio corner',objects:[obj(0,1,1),obj(2,5,5)],i:0,x:.1,y:1}];
let seed=20261030;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
for(let n=0;n<300;n++)scenarios.push({name:'random-'+n,objects:[obj(0,1,1),obj(1,3,3),obj(2,5,5)],i:n%3,x:-1+8*random(),y:-1+8*random()});
scenarios.push({name:'fixed cabinet overlap',objects:[obj(0,1,1),obj(2,5,5)],i:0,x:3,y:5.72});
const results=[];
for(const s of scenarios){const before=JSON.stringify(s.objects),p=context.nearestPlacement(s.objects,s.i,s.x,s.y,cfg);assert(p,s.name);assert.strictEqual(JSON.stringify(s.objects),before,'Projection must not mutate peers');const objects=structuredClone(s.objects);objects[s.i].x=p.x;objects[s.i].y=p.y;results.push({name:s.name,objects,placement:p});}
const shared=results.find(r=>r.name==='peer overlap with same-zone solution').objects;assert(shared.every(o=>Math.floor(o.x/2)===1&&Math.floor(o.y/2)===1),'Same-zone positions must remain allowed by geometry');
assert.strictEqual(context.nearestPlacement([obj(0,1,1)],0,NaN,1,cfg),null);
fs.mkdirSync('outputs/home_v5',{recursive:true});fs.writeFileSync('outputs/home_v5/placement_cases.json',JSON.stringify(results));
console.log('PASS:306 furnished-room projections, peer coordinates preserved, same-zone geometry allowed, nonfinite target rejected.');
