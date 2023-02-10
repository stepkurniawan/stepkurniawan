from docplex.mp.model import Model

mdl = Model(name="busLG")

# enable paralell processing
mdl.context.cplex_parameters.threads = 64
mdl.context.cplex_parameters.parallel = -1
mdl.context.solver.agent = "local"
mdl.context.solver.parallel = 4

import pandas as pd
import numpy as np

distance_adj_matrix_df= pd.read_excel('distance_adj_matrix_df_cplex_mini.xlsx')
time_adj_matrix_df= pd.read_excel('time_adj_matrix_df_cplex_mini.xlsx')
student_df_agg = pd.read_excel('student_df_agg_mini.xlsx')

adj_ind = time_adj_matrix_df.id.values.tolist()


#FIXED!!!!: the depot also needs to be included in the matrix
t=pd.DataFrame(time_adj_matrix_df.iloc[:,1:]) #preparing the matrix t (time adj)
t=t.transpose()
t.index = [adj_ind]
t.columns =[adj_ind]

d=pd.DataFrame(distance_adj_matrix_df.iloc[:,1:]) #preparing the matrix d (distance adj)
d=d.transpose()
d.columns =[adj_ind]
d.index = [adj_ind]

N=student_df_agg.iloc[:,0].values.tolist()
B=[5600,5606,5607,5608]
L=time_adj_matrix_df.id.values.tolist()
S = student_df_agg['school_name'].unique().tolist()
L_=L[1:]

#school of pupil
student_school_np=np.array([1,1,1,1,1,1,1,1,1,1])
s_n=pd.DataFrame(student_school_np,index=N,columns=S)


#set of bus stops of school
school_bus_stops_np=np.array([[0,0,0,0,0,1,0,0,0,0]])
I_s=pd.DataFrame(school_bus_stops_np,index=S,columns=L_)

#FIXED???: too many Bus stops in here in comparison to L_ & L... now L_ is all bus stops as we agreed (both pickup and school)
I_s_n=s_n.dot(I_s)
I_s_n = np.transpose(I_s_n)

school_max_transfers_np=np.array([[1]])
l_s_n=s_n.dot(school_max_transfers_np)
l_s_n = np.transpose(l_s_n)

Bus=((B[0],1,81,0.003),
     (B[1],1,81,0.003),
     (B[2],1,81,0.003),
     (B[3],1,81,0.003),
     )

Bus2=pd.DataFrame(Bus) #turning bus related parameters into dataframe
Bus2=Bus2.set_index(0)

c_b=Bus2.drop(Bus2.columns[[0,2]], axis=1) #bus related cost parameters with bus line as index
c_b= np.transpose(c_b)
q_b=Bus2.drop(Bus2.columns[[0,1]], axis=1)
q_fix = 40/60
gamma_up=15
gamma_low=5

school_wait_up_np=np.array([[1]])
school_wait_low_np=np.array([[1]])
tau_np=np.array([[1]])
w_up_list=list([30])
w_low_list=list([5])
tau_list = list([480])
w_up_sch_s=pd.DataFrame(school_wait_up_np,index=S,columns=w_up_list)
w_low_sch_s=pd.DataFrame(school_wait_low_np,index=S,columns=w_low_list)
tau_sch_s=pd.DataFrame(tau_np,index=S,columns=tau_list)

w_up=s_n.dot(w_up_sch_s)
w_up=np.transpose(w_up)
w_low=s_n.dot(w_low_sch_s)
w_low=np.transpose(w_low)
tau_s=s_n.dot(tau_sch_s)
tau_s=np.transpose(tau_s)


r_n = pd.read_excel('r_n.xlsx')
r_n.index= N
r_n = np.transpose(r_n)

max_tt_n =pd.read_excel('max_tt_n.xlsx') # max travel time
max_tt_n.index= N
max_tt_n = np.transpose(max_tt_n)

#
y_n = pd.read_excel('y_n.xlsx')
y_n.index = N
y_n = np.transpose(y_n)

# BINARY VARIABLES (only execute once!)
X = [(i, j, b) for i in L for j in L for b in B if i != j]
x = mdl.binary_var_dict(X, name='x')

V_1 = [(i_1, b_1, b_2) for i_1 in L for b_1 in B for b_2 in B if b_1 != b_2]
v = mdl.binary_var_dict(V_1, name='v')

M = [(n, i, j, b) for n in N for i in L for j in L for b in B if i != j]
m = mdl.binary_var_dict(M, name='m')

Z = [(n, b, i) for n in N for b in B for i in L]
z = mdl.binary_var_dict(Z, name='z')

# CONTINUOS VARIABLES
V_2 = [(i, b) for i in L for b in B]  # every stop for every bus
T_in = [(n, i) for n in N for i in L]

# arrival time of bus b at bus stop i
A = mdl.continuous_var_dict(V_2, ub=None, name='a')

# arrival time of pupil n at bus stop i
T = mdl.continuous_var_dict(T_in, ub=None, name='t')

# adjustable parameters
beta = 0.5
M_transfer = 10000

print(I_s_n[45].index[I_s_n[45] == 1].tolist())
print("Cplex is built")

# cost function, check
print("mdl.minimize")
mdl.minimize((1 - beta) * mdl.sum(
    x[i, j, b] * int(np.array(t[[i]].loc[[j]])) * q_fix + int(np.array(d[[i]].loc[[j]])) * x[i, j, b] * float(
        np.array(q_b.loc[[b]]))
    for i in L_
    for j in L_ if i != j
    for b in B)
             + (beta) * (0.1) * mdl.sum(m[n, i, j, b] * int(np.array(t[[i]].loc[[j]])) * int(np.array(r_n[n]))
                                        for i in L_
                                        for j in L_ if i != j
                                        for b in B
                                        for n in N))

print("A bus line must have a single origin at the virtual depot if the bus is used, check")
# mdl.add_constraints(x[0,i,b]>=mdl.sum(x[i,j,b] for j in L_ if j!=i)-mdl.sum(x[k,i,b] for k in L_ if k!=i) for b in B for i in L_)
mdl.add_constraints(mdl.sum(x[0, i, b] for i in L_) <= 1 for b in B)

# no cycles
# mdl.add_constraints(mdl.sum(x[j,i,b] for j in L_ if i!=j)+mdl.sum(x[i,j,b] for j in L_ if i!=j)<=1 for b in B for i in L_)

print("must return to virtual depot")
mdl.add_constraints(
    mdl.sum(x[j, i, b] for j in L if i != j) - mdl.sum(x[i, j, b] for j in L if i != j) == 0 for b in B for i in L_)

print("Every bus line may service each bus stop only once, check")
mdl.add_constraints(mdl.sum(x[i, j, b] for j in L_ if i != j) <= 1
                    for b in B
                    for i in L)

print("Students can only use existing bus lines, check")
mdl.add_constraints(m[n, i, j, b] <= x[i, j, b]
                    for i in L_
                    for j in L_ if i != j
                    for b in B
                    for n in N)

print(
    "If pupil n is assigned to bus stop i (yni = 1) which is not her destination bus stop i = isn she must leave the pickup busstop")

mdl.add_constraints(y_n[n].loc[i] <= mdl.sum(m[n, i, j, b] for j in L_ if j != i for b in B)
                    for n in N for i in y_n[n].index[y_n[n] == 1].tolist() if
                    i in I_s_n[n].index[I_s_n[n] == 0].tolist())

print("if pupil n arrives at bus stop and it is not her destination bus stop, she must travel on to another bus stop ")

mdl.add_constraints(
    mdl.sum(m[n, i, h, b] for i in L_ if h != i for b in B) <= mdl.sum(m[n, h, j, b] for j in L_ if h != j for b in B)
    for n in N for h in I_s_n[n].index[I_s_n[n] == 0].tolist())

# Every student (or sets of pupils) n N must arrive at exactly one bus stop for their destination school
# mdl.add_constraints(mdl.sum(m[n,j,i,b]
# for b in B
# for i in I_s_n[n].index[I_s_n[n]==1].tolist()
# for j in L_ if i!=j) == 1 for n in N)

print("A student can leave a bus stop only once, check")
mdl.add_constraints(mdl.sum(m[n, i, j, b]
                            for j in L_ if j != i
                            for b in B) <= 1
                    for i in L_
                    for n in N)

print("feasible transfers, check")
mdl.add_constraints(
    mdl.sum(m[n, j, i, b] for j in L_ if j != i) - mdl.sum(m[n, i, j, b] for j in L_ if j != i) <= z[n, b, i]
    for i in L_
    for n in N
    for b in B)

# mdl.add_constraints(mdl.sum(m[n,j,i,b]  for i in L_ for j in L_ if j!=i) >= mdl.sum(z[n,b,i]  for i in L_)
# for n in N
# for b in B)
#

print("Each pupil may only leave each bus b at most once, check")
mdl.add_constraints(mdl.sum(z[n, b, i] for i in L_) <= 1
                    for b in B
                    for n in N)

print(I_s_n[43].index[I_s_n[43] == 0])

print("Each student (or sets of students) n does not exceed the transfer limit of their school sn, check")
mdl.add_constraints(mdl.sum(z[n, b, i]
                            for b in B
                            for i in L_ if i in I_s_n[n].index[I_s_n[n] == 0].tolist()) <= int(
    np.array(l_s_n[n].index[l_s_n[n] == 1].tolist())) for n in N)
mdl.add_constraints(mdl.sum(z[n, b, i]
                            for b in B
                            for i in L_ if i in I_s_n[n].index[I_s_n[n] == 1].tolist()) == 1 for n in N)

print("definition of v[j,b1b2], check")
mdl.add_constraints(mdl.sum(m[n, i, j, b_1] for i in L_ if i != j) + mdl.sum(m[n, j, k, b_2] for k in L_ if k != j) <= (
            v[j, b_1, b_2] + 1)
                    for n in N
                    for j in L_
                    for b_1 in B
                    for b_2 in B if b_1 != b_2)

mdl.add_constraints(v[j, b_1, b_2] + v[j, b_2, b_1] <= 1
                    for j in L_
                    for b_1 in B
                    for b_2 in B if b_1 != b_2)

print(
    "If pupil n travels from i to j then her arrival time at j must be greater or equal to the arrival time at i plus the time needed to travel from i to j (tij), check")
mdl.add_constraints(
    (T[n, i] + int(np.array(t[[i]].loc[[j]])) - M_transfer * (1 - mdl.sum(m[n, i, j, b] for b in B))) <= T[n, j]
    for i in L_
    for j in L_ if j != i
    for n in N)


print("Pupil n must arrive at one bus stop for her destination school within the schools arrival time")

mdl.add_indicator_constraints(mdl.indicator_constraint(m[n, j, i, b],
                                                       int(np.array(tau_s[n].index[tau_s[n] == 1])) - int(
                                                           np.array(w_up[n].index[w_up[n] == 1])) <= T[n, i]) for n in N
                              for i in I_s_n[n].index[I_s_n[n] == 1].tolist() for j in L_ if j != i for b in B)
mdl.add_indicator_constraints(mdl.indicator_constraint(m[n, j, i, b],
                                                       int(np.array(tau_s[n].index[tau_s[n] == 1])) - int(
                                                           np.array(w_low[n].index[w_low[n] == 1])) >= T[n, i]) for n in
                              N for i in I_s_n[n].index[I_s_n[n] == 1].tolist() for j in L_ if j != i for b in B)

print(
    "If bus b travels from i to j then arrival time at j must be equal to travel time at i plus travel time from i to j")
mdl.add_constraints(A[j, b] <= A[i, b] + int(np.array(t[[i]].loc[[j]])) + M_transfer * (1 - x[i, j, b])
                    for i in L
                    for j in L_ if i != j
                    for b in B)
mdl.add_constraints(A[j, b] >= A[i, b] + int(np.array(t[[i]].loc[[j]])) - M_transfer * (1 - x[i, j, b])
                    for i in L
                    for j in L_ if i != j
                    for b in B)

print("If pupil n travels on bus b her arrival time at stop i has to be the same as of bus b check")
mdl.add_constraints(T[n, i] >= A[i, b] - M_transfer * (1 - mdl.sum(m[n, j, i, b] for j in L if j != i))
                    for i in L
                    for n in N
                    for b in B)
mdl.add_constraints(T[n, i] <= A[i, b] + M_transfer * (1 - mdl.sum(m[n, j, i, b] for j in L if j != i))
                    for i in L
                    for n in N
                    for b in B)

print("If pupil n changes from b1 to b2 at stop i then the arrival time of b ")
print(
    "cant be greater than of b plus max waiting time y and not lower than arrival time of b plus min waiting time y check")
mdl.add_constraints((A[i, b_1] + gamma_up + M_transfer * (1 - v[i, b_1, b_2])) >= A[i, b_2]
                    for i in L
                    for n in N
                    for b_1 in B
                    for b_2 in B if b_1 != b_2)
mdl.add_constraints(A[i, b_2] >= (A[i, b_1] + gamma_low - M_transfer * (1 - v[i, b_1, b_2]))
                    for i in L
                    for n in N
                    for b_1 in B
                    for b_2 in B if b_1 != b_2)

print("capacity constraint check")
mdl.add_constraints(mdl.sum(m[n, i, j, b] * int(np.array(r_n[n]))
                            for j in L_ if j != i
                            for n in N) <= int(np.array(c_b[b]))
                    for i in L_
                    for b in B)

# OPTIONAL: to run parallely with and without this constraint -> max_tt needs to be creared

mdl.add_constraints(mdl.sum(m[n,i,j,b]*int(np.array(t[[i]].loc[[j]]))
                             for i in L_
                             for j in L_ if i!=j
                             for b in B)<= int(np.array(max_tt_n[n].index[max_tt_n[n] == 1])) for n in N)

"""
# Conflict refiner

from docplex.mp.conflict_refiner import ConflictRefiner

cr = ConflictRefiner()
print("Conflict refiner created")
crr = cr.refine_conflict(mdl, display=True)
print("Conflict refiner refined")
cr.display_conflicts(crr) 
"""

print('first solution possible')
mdl.parameters.mip.limits.solutions = 1
solution=mdl.solve(log_output=True)
print(solution)

with open("solution.txt", "w") as solfile:
    solfile.write(mdl.solution.to_string())
    
print('solution under curent cost')
mdl.parameters.mip.tolerances.uppercutoff = 1384
solution2=mdl.solve(log_output=True)
print(solution2)

with open("solution2.txt", "w") as solfile:
    solfile.write(mdl.solution.to_string())