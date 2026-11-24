program bicluster_consensus
  use sklearn, only: dp,make_checkerboard,consensus_score
  implicit none
  real(dp), allocatable :: x(:,:)
  logical, allocatable :: rows(:,:),columns(:,:)
  call make_checkerboard([12,10],2,x,rows,columns,random_state=7)
  print *, 'Self consensus:',consensus_score(rows,columns,rows,columns)
  print *, 'One recovered bicluster:',consensus_score(rows,columns,rows(:1,:),columns(:1,:))
end program
