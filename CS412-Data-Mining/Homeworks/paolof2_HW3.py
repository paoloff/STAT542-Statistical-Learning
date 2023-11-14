def read_file(file_name):

   # Read file:
   f = open(file_name,'r')    
   T = []
   for line in f:
         T.append(line.split())
   f.close()

   # Remove first line
   min_supp = str(T[0][0])
   T.pop(0)
   return T, int(min_supp)

def find_frequent_items(T, min_supp):
   
   items = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
   dic = {}
   freq_items = {}
   
   for item in items:
      dic[item] = 0
      for transactions in T:
         if item in transactions:
           dic[item] += 1
      if dic[item] >= min_supp:
         freq_items[item] = dic[item]

   return freq_items

def count_pattern(T, pattern):

   c = 0

   for t in T:
      c1 = 1
      for item in pattern:
         if item in t:
            pass
         else:
            c1 = 0
            break
      c += c1

   return c

def no_infrequent_subset(candidate, old_list):

   for i in range(len(candidate)):
      subset_list = [c for c in candidate if c != candidate[i]]
      if subset_list not in old_list:
         return False
      else:
         pass

   return True

def new_list(current_list):

   new_list = []
   k = len(current_list[0])

   if k == 1:
      for i in range(len(current_list)):
         for j in range(i+1, len(current_list)):
            candidate = current_list[i] + current_list[j]
            if no_infrequent_subset(candidate, current_list):
               new_list.append(candidate)


   else:

      for i in range(len(current_list)):
         for j in range(i+1, len(current_list)):
            if current_list[i][0:k-1] == current_list[j][0:k-1] and current_list[i][k-1] != current_list[j][k-1]:
               candidate = current_list[i] + [current_list[j][k-1]]
               if no_infrequent_subset(candidate, current_list):
                  new_list.append(candidate)
      
   return new_list


def apriori(T, min_supp, freq_items):

   L1 = [[x] for x in freq_items.keys()]
   L = [L1]
   k=2

   current = L1

   while current != []:
      new = new_list(current)
      for pattern in new:
         if count_pattern(T, pattern) < min_supp:
            new.remove(pattern)

      current = new
      L.append(current)


   return L


def freq_pattern_mining(file_name):
   
   # Read file:
   f = open(file_name,'r')    
   T = []
   for line in f:
         T.append(line.split())
   f.close()

   # Remove first line
   T.pop(0)

   # Count the support of D and DF:
   cd = 0
   cdf = 0
   for transactions in T:
       if 'D' in transactions:
         cd += 1
         ind = transactions.index('D') + 1
         if ind < len(transactions):
            if transactions[ind] == 'F':
               cdf += 1

   # Count the support of a list with 4 letters (eg BDEF):
   cl = 0
   l = ['B', 'D', 'E', 'F']
   for transactions in T:
       if l[0] in transactions:
         ind = transactions.index(l[0]) + 1
         if ind < len(transactions) - 3:
            if transactions[ind:ind+3] == l[1:3]:
               cl += 1
