`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/01/19 15:02:12
// Design Name: 
// Module Name: TOP
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Systolic_array(
	input					clk,
	input					rst_n,

	input					ac_valid,
	input	[31 : 0]		activate,
	input					we_valid,
	input	[127 : 0]		weight,

	input 	[12 : 0]		matmul_param,

	input					relu,

	output reg	[31 : 0]	matmul_data
    );

wire [7 : 0] activate_00;
wire [7 : 0] activate_01;
wire [7 : 0] activate_02;
wire [7 : 0] activate_03;

wire [7 : 0] activate_10;
wire [7 : 0] activate_11;
wire [7 : 0] activate_12;
wire [7 : 0] activate_13;

wire [7 : 0] activate_20;
wire [7 : 0] activate_21;
wire [7 : 0] activate_22;
wire [7 : 0] activate_23;

wire [7 : 0] activate_30;
wire [7 : 0] activate_31;
wire [7 : 0] activate_32;
wire [7 : 0] activate_33;

wire [15 : 0] prod_00;
wire [15 : 0] prod_01;
wire [15 : 0] prod_02;
wire [15 : 0] prod_03;

wire [15 : 0] prod_10;
wire [15 : 0] prod_11;
wire [15 : 0] prod_12;
wire [15 : 0] prod_13;

wire [15 : 0] prod_20;
wire [15 : 0] prod_21;
wire [15 : 0] prod_22;
wire [15 : 0] prod_23;

wire [15 : 0] prod_30;
wire [15 : 0] prod_31;
wire [15 : 0] prod_32;
wire [15 : 0] prod_33;

wire [21 : 0] sum_00;
wire [21 : 0] sum_01;
wire [21 : 0] sum_02;
wire [21 : 0] sum_03;

wire [21 : 0] sum_10;
wire [21 : 0] sum_11;
wire [21 : 0] sum_12;
wire [21 : 0] sum_13;

wire [21 : 0] sum_20;
wire [21 : 0] sum_21;
wire [21 : 0] sum_22;
wire [21 : 0] sum_23;

wire [21 : 0] sum_30;
wire [21 : 0] sum_31;
wire [21 : 0] sum_32;
wire [21 : 0] sum_33;

wire [21 : 0] sum_0;
wire [21 : 0] sum_1;
wire [21 : 0] sum_2;
wire [21 : 0] sum_3;

reg we_valid_00;
reg we_valid_01;
reg we_valid_02;
reg we_valid_03;

reg we_valid_10;
reg we_valid_11;
reg we_valid_12;
reg we_valid_13;

reg we_valid_20;
reg we_valid_21;
reg we_valid_22;
reg we_valid_23;

reg we_valid_30;
reg we_valid_31;
reg we_valid_32;
reg we_valid_33;

reg [3 : 0] cnt;

assign sum_0 = ((sum_03 + prod_03) * matmul_param[7 : 0]) >> matmul_param[12 : 8];
assign sum_1 = ((sum_13 + prod_13) * matmul_param[7 : 0]) >> matmul_param[12 : 8];
assign sum_2 = ((sum_23 + prod_23) * matmul_param[7 : 0]) >> matmul_param[12 : 8];
assign sum_3 = ((sum_33 + prod_33) * matmul_param[7 : 0]) >> matmul_param[12 : 8];



always @(negedge rst_n or posedge clk)
begin
	if(!rst_n)
	begin
		cnt <= 3'b0;
	end
	else if(we_valid && cnt == 3'b0)
	begin
		cnt <= 3'b1;
	end
	else if(we_valid)
	begin
		cnt <= cnt + 1;
	end
	else if(!we_valid)
	begin
		cnt <= 3'b0;
	end
	else
	begin
		
	end
end

always @(negedge rst_n or posedge clk)
begin
	if(!rst_n)
	begin
		we_valid_00 <= 1'b0;
		we_valid_01 <= 1'b0;
		we_valid_02 <= 1'b0;
		we_valid_03 <= 1'b0;
		we_valid_10 <= 1'b0;
		we_valid_11 <= 1'b0;
		we_valid_12 <= 1'b0;
		we_valid_13 <= 1'b0;
		we_valid_20 <= 1'b0;
		we_valid_21 <= 1'b0;
		we_valid_22 <= 1'b0;
		we_valid_23 <= 1'b0;
		we_valid_30 <= 1'b0;
		we_valid_31 <= 1'b0;
		we_valid_32 <= 1'b0;
		we_valid_33 <= 1'b0;
	end
	else if(cnt == 3'd1)
	begin
		we_valid_00 <= 1'b1;
		we_valid_01 <= 1'b1;
		we_valid_02 <= 1'b1;
		we_valid_03 <= 1'b1;
	end
	else if(cnt == 3'd2)
	begin
		we_valid_10 <= 1'b1;
		we_valid_11 <= 1'b1;
		we_valid_12 <= 1'b1;
		we_valid_13 <= 1'b1;
	end
	else if(cnt == 3'd3)
	begin
		we_valid_20 <= 1'b1;
		we_valid_21 <= 1'b1;
		we_valid_22 <= 1'b1;
		we_valid_23 <= 1'b1;
	end
	else if(cnt == 3'd4)
	begin
		we_valid_30 <= 1'b1;
		we_valid_31 <= 1'b1;
		we_valid_32 <= 1'b1;
		we_valid_33 <= 1'b1;
	end
	else
	begin
		
	end
end

always @(negedge rst_n or posedge clk)
begin
	if(!rst_n)
	begin
		matmul_data <= 32'b0;
	end
	else if(relu)
	begin
		matmul_data[7 : 0]   <= sum_0[21] ? 8'b0 : sum_0[7 : 0];
		matmul_data[15 : 8]  <= sum_1[21] ? 8'b0 : sum_1[7 : 0];
		matmul_data[23 : 16] <= sum_2[21] ? 8'b0 : sum_2[7 : 0];
		matmul_data[31 : 24] <= sum_3[21] ? 8'b0 : sum_3[7 : 0];
	end
	else if(!relu)
	begin
		matmul_data[7 : 0]   <= sum_0[7 : 0];
		matmul_data[15 : 8]  <= sum_1[7 : 0];
		matmul_data[23 : 16] <= sum_2[7 : 0];
		matmul_data[31 : 24] <= sum_3[7 : 0];
	end
	else 
	begin
		
	end
end

PE PE_00
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_00),
		.weight       (weight[7 : 0]),
		.ac_valid     (ac_valid),
		.activate     (activate[7 : 0]),
		.prod_in      (16'b0),
		.sum_in       (22'b0),
		.prod_out     (prod_00),
		.activate_out (activate_30),
		.sum_out      (sum_00)
	);

PE PE_01
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_01),
		.weight       (weight[15 : 8]),
		.ac_valid     (ac_valid),
		.activate     (activate[15 : 8]),
		.prod_in      (16'b0),
		.sum_in       (22'b0),
		.prod_out     (prod_10),
		.activate_out (activate_20),
		.sum_out      (sum_10)
	);

PE PE_02
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_02),
		.weight       (weight[23 : 16]),
		.ac_valid     (ac_valid),
		.activate     (activate[23 : 16]),
		.prod_in      (16'b0),
		.sum_in       (22'b0),
		.prod_out     (prod_20),
		.activate_out (activate_10),
		.sum_out      (sum_20)
	);

PE PE_03
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_03),
		.weight       (weight[31 : 24]),
		.ac_valid     (ac_valid),
		.activate     (activate[31 : 24]),
		.prod_in      (16'b0),
		.sum_in       (22'b0),
		.prod_out     (prod_30),
		.activate_out (activate_00),
		.sum_out      (sum_30)
	);

PE PE_10
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_10),
		.weight       (weight[7 : 0]),
		.ac_valid     (ac_valid),
		.activate     (activate_20),
		.prod_in      (prod_00),
		.sum_in       (sum_00),
		.prod_out     (prod_01),
		.activate_out (activate_21),
		.sum_out      (sum_01)
	);

PE PE_11
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_11),
		.weight       (weight[15 : 8]),
		.ac_valid     (ac_valid),
		.activate     (activate_10),
		.prod_in      (prod_10),
		.sum_in       (sum_10),
		.prod_out     (prod_11),
		.activate_out (activate_11),
		.sum_out      (sum_11)
	);

PE PE_12
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_12),
		.weight       (weight[23 : 16]),
		.ac_valid     (ac_valid),
		.activate     (activate_00),
		.prod_in      (prod_20),
		.sum_in       (sum_20),
		.prod_out     (prod_21),
		.activate_out (activate_01),
		.sum_out      (sum_21)
	);

PE PE_13
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_13),
		.weight       (weight[31 : 24]),
		.ac_valid     (ac_valid),
		.activate     (activate_30),
		.prod_in      (prod_30),
		.sum_in       (sum_30),
		.prod_out     (prod_31),
		.activate_out (activate_31),
		.sum_out      (sum_31)
	);

PE PE_20
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_20),
		.weight       (weight[7 : 0]),
		.ac_valid     (ac_valid),
		.activate     (activate_11),
		.prod_in      (prod_01),
		.sum_in       (sum_01),
		.prod_out     (prod_02),
		.activate_out (activate_12),
		.sum_out      (sum_02)
	);

PE PE_21
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_21),
		.weight       (weight[15 : 8]),
		.ac_valid     (ac_valid),
		.activate     (activate_01),
		.prod_in      (prod_11),
		.sum_in       (sum_11),
		.prod_out     (prod_12),
		.activate_out (activate_02),
		.sum_out      (sum_12)
	);

PE PE_22
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_22),
		.weight       (weight[23 : 16]),
		.ac_valid     (ac_valid),
		.activate     (activate_31),
		.prod_in      (prod_21),
		.sum_in       (sum_21),
		.prod_out     (prod_22),
		.activate_out (activate_32),
		.sum_out      (sum_22)
	);

PE PE_23
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_23),
		.weight       (weight[31 : 24]),
		.ac_valid     (ac_valid),
		.activate     (activate_21),
		.prod_in      (prod_31),
		.sum_in       (sum_31),
		.prod_out     (prod_32),
		.activate_out (activate_22),
		.sum_out      (sum_32)
	);

PE PE_30
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_30),
		.weight       (weight[7 : 0]),
		.ac_valid     (ac_valid),
		.activate     (activate_02),
		.prod_in      (prod_02),
		.sum_in       (sum_02),
		.prod_out     (prod_03),
		.activate_out (activate_03),
		.sum_out      (sum_03)
	);

PE PE_31
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_31),
		.weight       (weight[15 : 8]),
		.ac_valid     (ac_valid),
		.activate     (activate_32),
		.prod_in      (prod_12),
		.sum_in       (sum_12),
		.prod_out     (prod_13),
		.activate_out (activate_33),
		.sum_out      (sum_13)
	);

PE PE_32
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_32),
		.weight       (weight[23 : 16]),
		.ac_valid     (ac_valid),
		.activate     (activate_22),
		.prod_in      (prod_22),
		.sum_in       (sum_22),
		.prod_out     (prod_23),
		.activate_out (activate_23),
		.sum_out      (sum_23)
	);

PE PE_33
	(
		.clk          (clk),
		.rst_n        (rst_n),
		.we_valid     (we_valid_33),
		.weight       (weight[31 : 24]),
		.ac_valid     (ac_valid),
		.activate     (activate_12),
		.prod_in      (prod_32),
		.sum_in       (sum_32),
		.prod_out     (prod_33),
		.activate_out (activate_13),
		.sum_out      (sum_33)
	);

endmodule
