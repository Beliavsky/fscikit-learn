program test_ward_tree
  use sklearn,only:dp,ward_tree,hierarchy_result,csr_from_dense,csr_from_arrays,csr_matrix
  use sklearn_common,only:require
  use,intrinsic::ieee_exceptions
  implicit none
  real(dp)::x(4),g(4,4)
  type(hierarchy_result)::tree,other
  type(csr_matrix)::sparse
  real(dp),allocatable::large(:),values(:)
  integer,allocatable::ptr(:),columns(:)
  logical::flags(size(ieee_all)),after(size(ieee_all)),modes(size(ieee_all)),after_modes(size(ieee_all))
  integer::i,pos,n
  x=[0._dp,1._dp,4._dp,5._dp];g=1
  tree=ward_tree(x,return_distance=.true.)
  call require(all(tree%children(:,1)==[1,3,5]).and.all(tree%children(:,2)==[2,4,6]),'Ward source golden tree')
  call require(maxval(abs(tree%distances-[1._dp,1._dp,sqrt(32._dp)]))<1.e-14_dp,'Ward golden heights')
  call require(size(tree%parents)==0.and..not.tree%structured,'unstructured parents represent source None')
  other=ward_tree(reshape(x,[4,1]),return_distance=.true.)
  call require(all(other%children==tree%children),'matrix/vector unstructured Ward')
  other=ward_tree(x,g,return_distance=.true.)
  call require(all(other%children==tree%children),'real connectivity vector overload')
  call require(all(other%parents==[5,5,6,6,7,7,7]),'full structured parents')
  other=ward_tree(x,g/=0,return_distance=.true.)
  call require(all(other%children==tree%children),'logical connectivity vector overload')
  other=ward_tree(x,csr_from_dense(g),n_clusters=2,return_distance=.true.)
  call require(size(other%children,1)==2.and.all(other%parents==[5,5,6,6,5,6]),'partial CSR vector forest')
  other=ward_tree(reshape(x,[4,1]),g,n_clusters=4)
  call require(size(other%children,1)==0.and.all(other%parents==[1,2,3,4]),'unmerged real matrix forest')
  other=ward_tree(reshape(x,[4,1]),g/=0,n_clusters=2)
  call require(size(other%distances)==0.and.size(other%children,1)==2,'logical matrix distance omission')
  g=0
  other=ward_tree(reshape(x,[4,1]),csr_from_dense(g),return_distance=.true.)
  call require(other%n_connected_components==4.and.other%connectivity_completed,'disconnected completion metadata')
  call require(all(other%children==tree%children),'component completion creates all required closest edges')
  call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  other=ward_tree(x,return_distance=.true.)
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after).and.all(modes.eqv.after_modes),'Ward IEEE state preservation')
  n=32768;allocate(large(n),ptr(n+1),columns(2*n-2),values(2*n-2))
  large=[(real(i,dp),i=1,n)];pos=1;ptr(1)=1;values=1
  do i=1,n
    if(i>1)then
      columns(pos)=i-1;pos=pos+1
    end if
    if(i<n)then
      columns(pos)=i+1;pos=pos+1
    end if
    ptr(i+1)=pos
  end do
  sparse=csr_from_arrays(n,n,ptr,columns,values)
  other=ward_tree(large,sparse,n_clusters=n)
  call require(size(other%children,1)==0.and.size(other%parents)==n,'large sparse partial forest')
  call require(other%n_connected_components==1.and.all(other%parents==[(i,i=1,n)]),'large sparse forest roots')
  print *,'Ward tree tests passed; avoided an 8 GiB dense connectivity matrix'
end program
