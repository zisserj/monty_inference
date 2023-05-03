
// const fs = require('fs');

var obsData = [
	{guess: 'd1', left: 'd2', choose: 'd2', win: true},
	{guess: 'd2', left: 'd3', choose: 'd3', win: false}
  ]
// 'GiCjW'
const obsStr = ["G2L3C3W","G1L2C1L"]


  
const obsRE = /G(\d+)L(\d+)C(\d+)(W|L)/
const obsMatch = obsStr.forEach(function(o) {
var match = o.match(obsRE)
obsData.push({guess: 'd'+match[1], left: 'd'+match[2], choose: 'd'+match[3], win: 'W' == match[4]})
});

console.log(obsData)

