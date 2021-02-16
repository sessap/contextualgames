

% import Table
Table = table2array(SiouxFallstrips); 

nodes = size(Table,1)/2;
OD_matrix = NaN*ones(nodes,nodes);

for or = 1:nodes
    for des = 1:nodes
        OD_matrix(or, des) =  Table(2*or, 2*des);       
    end
end

dlmwrite('SiouxFalls_OD_matrix.txt', OD_matrix)
