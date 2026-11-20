// OpenCV.js (Emscripten's embind) builds its call wrappers by compiling
// JavaScript source at run time (`new Function`), which a strict
// Content-Security-Policy forbids. This rewrites the two places that do so
// as plain closures with the same behaviour, so the site needs no
// 'unsafe-eval' anywhere.
//
//   import { patchOpenCv } from './patch-opencv.mjs'; const text = patchOpenCv(source);

function replaceBetween(text, start, end, replacement) {
  const a = text.indexOf(start);
  if (a < 0) throw new Error(`patch-opencv: start not found: ${start.slice(0, 60)}`);
  const b = text.indexOf(end, a);
  if (b < 0) throw new Error(`patch-opencv: end not found: ${end.slice(0, 60)}`);
  if (text.indexOf(start, a + 1) >= 0) throw new Error('patch-opencv: start found twice');
  return text.slice(0, a) + replacement + text.slice(b + end.length);
}

// craftInvokerFunction: the wrapper around every bound C++ function and method.
const CRAFT = `function craftInvokerFunction(humanName,argTypes,classType,cppInvokerFunc,cppTargetFunc,isAsync){
var argCount=argTypes.length;
if(argCount<2){throwBindingError("argTypes array size mismatch! Must at least get return value and 'this' types!")}
var isClassMethodFunc=argTypes[1]!==null&&classType!==null;
var needsDestructorStack=usesDestructorStack(argTypes);
var returns=argTypes[0].name!=="void";
var expected=argCount-2;
var first=isClassMethodFunc?2:1;
var invokerFn=function(...args){
if(args.length!==expected){throwBindingError("function "+humanName+" called with "+args.length+" arguments, expected "+expected)}
var destructors=needsDestructorStack?[]:null;
var wired=[cppTargetFunc];
var thisWired;
if(isClassMethodFunc){thisWired=argTypes[1]["toWireType"](destructors,this);wired.push(thisWired)}
for(var i=0;i<expected;++i){wired.push(argTypes[i+2]["toWireType"](destructors,args[i]))}
var rv=cppInvokerFunc(...wired);
if(needsDestructorStack){runDestructors(destructors)}
else{for(var i=isClassMethodFunc?1:2;i<argCount;++i){if(argTypes[i].destructorFunction!==null){argTypes[i].destructorFunction(i===1?thisWired:wired[i-2+first])}}}
if(returns){return argTypes[0]["fromWireType"](rv)}
};
return createNamedFunction(humanName,invokerFn)}`;

// __emval_get_method_caller: calls from C++ into JavaScript objects.
const METHOD_CALLER = `var __emval_get_method_caller=(argCount,argTypes,kind)=>{
var types=emval_lookupTypes(argCount,argTypes);var retType=types.shift();argCount--;
var offsets=[];var offset=0;for(var i=0;i<argCount;++i){offsets.push(offset);offset+=types[i]["argPackAdvance"]}
var invokerFunction=function(obj,func,destructorsRef,args){
var values=kind===0?[obj]:[];
for(var i=0;i<argCount;++i){values.push(types[i].readValueFromPointer(args+offsets[i]))}
var rv=kind===1?new func(...values.slice(0)):func.call(...values);
if(!retType.isVoid){return emval_returnValue(retType,destructorsRef,rv)}};
var functionName=\`methodCaller<(\${types.map(t=>t.name).join(", ")}) => \${retType.name}>\`;
return emval_addMethodCaller(createNamedFunction(functionName,invokerFunction))};`;

export function patchOpenCv(source) {
  let text = replaceBetween(source,
    'function craftInvokerFunction(humanName,argTypes,classType,cppInvokerFunc,cppTargetFunc,isAsync){',
    'return createNamedFunction(humanName,invokerFn)}', CRAFT);
  text = replaceBetween(text, 'var __emval_get_method_caller=(argCount,argTypes,kind)=>{',
    'return emval_addMethodCaller(createNamedFunction(functionName,invokerFunction))};', METHOD_CALLER);
  for (const forbidden of ['newFunc(Function', 'new Function', 'eval(']) {
    if (text.includes(forbidden)) throw new Error(`patch-opencv: "${forbidden}" is still in OpenCV.js`);
  }
  return text;
}
