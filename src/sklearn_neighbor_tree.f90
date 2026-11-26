! sklearn 1.6.1 neighbors/_binary_tree.pxi.tp and _kd_tree.pyx.tp (BSD).
! Native owned storage and scalar query adapters are MIT; see notices.
module sklearn_neighbor_tree
 use sklearn_common,only:dp,require,check_x
 use sklearn_tree_partition,only:partition_node
 use sklearn_neighbor_selection,only:neighbor_heap_push,neighbor_simultaneous_sort
 use sklearn_pairwise_reductions,only:row_distance
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::neighbor_kd_tree
 type::neighbor_kd_tree
  private
  real(dp),allocatable::x(:,:),bounds(:,:,:)
  integer,allocatable::order(:),spans(:,:)
  integer::kind=3
  real(dp)::power=2
 contains
  procedure::fit=>tree_fit
  procedure::query=>tree_query
  procedure::query_radius=>tree_radius
  procedure::get_arrays=>tree_arrays
 end type
contains
 subroutine tree_fit(self,x,metric,p,leaf_size)
  class(neighbor_kd_tree),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  integer,optional,intent(in)::leaf_size
  character(16)::name
  integer::leaf,n,ratio,levels,nodes,i
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call check_x(x);name='euclidean';if(present(metric))name=metric
  self%power=2;if(present(p))self%power=p
  call require(.not.ieee_is_nan(self%power),'KD tree p is NaN')
  call require(self%power>0,'KD tree p must be positive')
  select case(name)
  case('euclidean','l2');self%kind=3;self%power=2
  case('manhattan','cityblock','l1');self%kind=4;self%power=1
  case('chebyshev','infinity');self%kind=5;self%power=ieee_value(0._dp,ieee_positive_inf)
  case('minkowski')
   call require(self%power>=1,'KD tree Minkowski p must be at least one')
   self%kind=6
   if(self%power==1)self%kind=4
   if(self%power==2)self%kind=3
   if(.not.ieee_is_finite(self%power))self%kind=5
  case default;call require(.false.,'unsupported KD tree metric')
  end select
  leaf=30;if(present(leaf_size))leaf=leaf_size
  call require(leaf>0,'KD tree leaf size must be positive')
  n=size(x,1);levels=1;ratio=max(1,(n-1)/leaf)
  do while(ratio>=2);ratio=ratio/2;levels=levels+1;end do
  call require(levels<30,'KD tree node capacity exceeded');nodes=2**levels-1
  self%x=x
  if(allocated(self%spans))deallocate(self%spans,self%bounds,self%order)
  allocate(self%spans(nodes,3),self%bounds(2,nodes,size(x,2)),self%order(n));self%order=[(i,i=1,n)]
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call grow(1,1,n)
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 contains
  recursive subroutine grow(node,first,last)
   integer,intent(in)::node,first,last
   integer::feature,middle
   self%bounds(1,node,:)=minval(self%x(self%order(first:last),:),dim=1)
   self%bounds(2,node,:)=maxval(self%x(self%order(first:last),:),dim=1)
   self%spans(node,:)=[first,last,merge(1,0,2*node>nodes)]
   if(2*node>nodes)return
   feature=maxloc(self%bounds(2,node,:)-self%bounds(1,node,:),dim=1);middle=first+(last-first+1)/2
   call partition_node(self%x(:,feature),self%order(first:last),middle-first+1)
   call grow(2*node,first,middle-1);call grow(2*node+1,middle,last)
  end subroutine
 end subroutine
 subroutine tree_arrays(self,indices,spans,bounds)
  class(neighbor_kd_tree),intent(in)::self
  integer,allocatable,intent(out)::indices(:),spans(:,:)
  real(dp),allocatable,intent(out)::bounds(:,:,:)
  call require(allocated(self%order),'KD tree is not fitted')
  indices=self%order;spans=self%spans;bounds=self%bounds
 end subroutine
 real(dp)function actual_distance(self,reduced)result(value)
  class(neighbor_kd_tree),intent(in)::self
  real(dp),intent(in)::reduced
  value=reduced
  if(self%kind==3)value=sqrt(reduced)
  if(self%kind==6)value=reduced**(1._dp/self%power)
 end function
 real(dp)function reduced_radius(self,radius)result(value)
  class(neighbor_kd_tree),intent(in)::self
  real(dp),intent(in)::radius
  value=radius
  if(self%kind==3)value=radius*radius
  if(self%kind==6)value=radius**self%power
 end function
 subroutine node_bounds(self,node,point,lower,upper)
  class(neighbor_kd_tree),intent(in)::self
  integer,intent(in)::node
  real(dp),intent(in)::point(:)
  real(dp),intent(out)::lower,upper
  real(dp)::lo,hi,d
  integer::j
  lower=0;upper=0
  do j=1,size(point)
   lo=self%bounds(1,node,j)-point(j);hi=point(j)-self%bounds(2,node,j)
   d=(lo+abs(lo))+(hi+abs(hi))
   if(self%kind==5)then
    ! Source fmax ignores one NaN operand.
    if(.not.ieee_is_nan(d))lower=max(lower,.5_dp*d)
    if(.not.ieee_is_nan(lo))upper=max(upper,abs(lo))
    if(.not.ieee_is_nan(hi))upper=max(upper,abs(hi))
   else
    lower=lower+(.5_dp*d)**self%power
    upper=upper+max(abs(lo),abs(hi))**self%power
   end if
  end do
 end subroutine
 subroutine check_query(self,point)
  class(neighbor_kd_tree),intent(in)::self
  real(dp),intent(in)::point(:)
  call require(allocated(self%order),'KD tree is not fitted')
  call require(size(point)==size(self%x,2),'KD tree query feature count')
  call require(all(ieee_is_finite(point)),'KD tree query must be finite')
 end subroutine
 subroutine tree_query(self,point,k,distances,indices)
  class(neighbor_kd_tree),intent(in)::self
  real(dp),intent(in)::point(:)
  integer,intent(in)::k
  real(dp),allocatable,intent(out)::distances(:)
  integer,allocatable,intent(out)::indices(:)
  real(dp)::lower,upper
  integer::j
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call check_query(self,point);call require(k>=1.and.k<=size(self%x,1),'KD tree query k out of range')
  allocate(distances(k),indices(k));distances=ieee_value(0._dp,ieee_positive_inf);indices=1
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call node_bounds(self,1,point,lower,upper);call visit(1,lower)
  call neighbor_simultaneous_sort(distances,indices)
  do j=1,k;distances(j)=actual_distance(self,distances(j));end do
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 contains
  recursive subroutine visit(node,lower)
   integer,intent(in)::node
   real(dp),intent(in)::lower
   real(dp)::left,right,unused,value
   integer::at,id
   if(lower>distances(1))return
   if(self%spans(node,3)==1)then
    do at=self%spans(node,1),self%spans(node,2)
     id=self%order(at);value=row_distance(point,self%x(id,:),self%kind,self%power,.false.,.false.,.false.)
     call neighbor_heap_push(distances,indices,value,id)
    end do
   else
    call node_bounds(self,2*node,point,left,unused);call node_bounds(self,2*node+1,point,right,unused)
    if(left<=right)then
     call visit(2*node,left);call visit(2*node+1,right)
    else
     call visit(2*node+1,right);call visit(2*node,left)
    end if
   end if
  end subroutine
 end subroutine
 subroutine tree_radius(self,point,radius,distances,indices)
  class(neighbor_kd_tree),intent(in)::self
  real(dp),intent(in)::point(:),radius
  real(dp),allocatable,intent(out)::distances(:)
  integer,allocatable,intent(out)::indices(:)
  integer::used
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call check_query(self,point);call require(.not.ieee_is_nan(radius),'KD tree radius is NaN')
  call require(radius>=0,'KD tree radius must be nonnegative')
  allocate(distances(size(self%x,1)),indices(size(self%x,1)));used=0
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call visit(1);distances=distances(:used);indices=indices(:used)
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 contains
  recursive subroutine visit(node)
   integer,intent(in)::node
   real(dp)::lo,hi,cutoff,value
   integer::at,id
   call node_bounds(self,node,point,lo,hi)
   if(self%kind/=5)then
    lo=lo**(1._dp/self%power);hi=hi**(1._dp/self%power)
   end if
   if(lo>radius)return
   if(hi<=radius)then
    do at=self%spans(node,1),self%spans(node,2)
     id=self%order(at);used=used+1;indices(used)=id
     distances(used)=actual_distance(self,row_distance(point,self%x(id,:),self%kind,self%power,.false.,.false.,.false.))
    end do
   else if(self%spans(node,3)==1)then
    cutoff=reduced_radius(self,radius)
    do at=self%spans(node,1),self%spans(node,2)
     id=self%order(at);value=row_distance(point,self%x(id,:),self%kind,self%power,.false.,.false.,.false.)
     if(value<=cutoff)then
      used=used+1;indices(used)=id;distances(used)=actual_distance(self,value)
     end if
    end do
   else
    call visit(2*node);call visit(2*node+1)
   end if
  end subroutine
 end subroutine
end module
