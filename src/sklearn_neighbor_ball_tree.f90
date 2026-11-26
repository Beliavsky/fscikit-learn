! sklearn 1.6.1 neighbors/_binary_tree.pxi.tp and _ball_tree.pyx.tp (BSD).
! Native owned storage and scalar query adapters are MIT; see notices.
module sklearn_neighbor_ball_tree
 use sklearn_common,only:dp,require,check_x
 use sklearn_tree_partition,only:partition_node
 use sklearn_neighbor_selection,only:neighbor_heap_push,neighbor_simultaneous_sort
 use sklearn_pairwise_reductions,only:row_distance,metric_kind,extended_distance
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::neighbor_ball_tree
 type::neighbor_ball_tree
  private
  real(dp),allocatable::x(:,:),centers(:,:),radii(:)
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
  class(neighbor_ball_tree),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  integer,optional,intent(in)::leaf_size
  character(16)::name
  integer::leaf,n,ratio,levels,nodes,i
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call check_x(x);name='euclidean';if(present(metric))name=metric
  self%power=2;if(present(p))self%power=p
  call require(.not.ieee_is_nan(self%power),'Ball tree p is NaN')
  call require(self%power>0,'Ball tree p must be positive')
  select case(name)
  case('euclidean','l2');self%kind=3;self%power=2
  case('manhattan','cityblock','l1');self%kind=4;self%power=1
  case('chebyshev','infinity');self%kind=5;self%power=ieee_value(0._dp,ieee_positive_inf)
  case('minkowski')
   self%kind=6
   if(self%power==1)self%kind=4
   if(self%power==2)self%kind=3
   if(.not.ieee_is_finite(self%power))self%kind=5
  case('haversine','braycurtis','canberra','hamming','dice','jaccard', &
   'rogerstanimoto','russellrao','sokalmichener','sokalsneath')
   self%kind=metric_kind(name)
   if(self%kind==9)call require(size(x,2)==2,'Ball tree haversine requires two features')
  case default;call require(.false.,'unsupported Ball tree metric')
  end select
  leaf=30;if(present(leaf_size))leaf=leaf_size
  call require(leaf>0,'Ball tree leaf size must be positive')
  n=size(x,1);levels=1;ratio=max(1,(n-1)/leaf)
  do while(ratio>=2);ratio=ratio/2;levels=levels+1;end do
  call require(levels<30,'Ball tree node capacity exceeded');nodes=2**levels-1
  self%x=x
  if(allocated(self%spans))deallocate(self%spans,self%centers,self%radii,self%order)
  allocate(self%spans(nodes,3),self%centers(nodes,size(x,2)),self%radii(nodes),self%order(n));self%order=[(i,i=1,n)]
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call grow(1,1,n)
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 contains
  recursive subroutine grow(node,first,last)
   integer,intent(in)::node,first,last
   integer::feature,middle,at,id,j
   real(dp)::low(size(x,2)),high(size(x,2)),radius,value
   self%centers(node,:)=0
   do at=first,last
    id=self%order(at)
    do j=1,size(x,2)
     self%centers(node,j)=self%centers(node,j)+self%x(id,j)
    end do
   end do
   self%centers(node,:)=self%centers(node,:)/(last-first+1)
   radius=0
   do at=first,last
    value=point_distance(self,self%centers(node,:),self%x(self%order(at),:))
    if(.not.ieee_is_nan(value))radius=max(radius,value)
   end do
   self%radii(node)=actual_distance(self,radius)
   self%spans(node,:)=[first,last,merge(1,0,2*node>nodes)]
   if(2*node>nodes)return
   low=minval(self%x(self%order(first:last),:),dim=1)
   high=maxval(self%x(self%order(first:last),:),dim=1)
   feature=maxloc(high-low,dim=1);middle=first+(last-first+1)/2
   call partition_node(self%x(:,feature),self%order(first:last),middle-first+1)
   call grow(2*node,first,middle-1);call grow(2*node+1,middle,last)
  end subroutine
 end subroutine
 subroutine tree_arrays(self,indices,spans,centers,radii)
  class(neighbor_ball_tree),intent(in)::self
  integer,allocatable,intent(out)::indices(:),spans(:,:)
  real(dp),allocatable,intent(out)::centers(:,:),radii(:)
  call require(allocated(self%order),'Ball tree is not fitted')
  indices=self%order;spans=self%spans;centers=self%centers;radii=self%radii
 end subroutine
 real(dp)function actual_distance(self,reduced)result(value)
  class(neighbor_ball_tree),intent(in)::self
  real(dp),intent(in)::reduced
  value=reduced
  if(self%kind==3)value=sqrt(reduced)
  if(self%kind==6)value=reduced**(1._dp/self%power)
  if(self%kind==9)value=2*asin(sqrt(reduced))
 end function
 real(dp)function reduced_radius(self,radius)result(value)
  class(neighbor_ball_tree),intent(in)::self
  real(dp),intent(in)::radius
  value=radius
  if(self%kind==3)value=radius*radius
  if(self%kind==6)value=radius**self%power
  if(self%kind==9)value=sin(.5_dp*radius)**2
 end function
 subroutine node_bounds(self,node,point,lower,upper)
  class(neighbor_ball_tree),intent(in)::self
  integer,intent(in)::node
  real(dp),intent(in)::point(:)
  real(dp),intent(out)::lower,upper
  real(dp)::distance,delta
  distance=actual_distance(self,point_distance(self,point,self%centers(node,:)))
  delta=distance-self%radii(node);lower=0
  if(.not.ieee_is_nan(delta))lower=max(0._dp,delta)
  upper=distance+self%radii(node)
 end subroutine
 real(dp)function point_distance(self,u,v)result(value)
  class(neighbor_ball_tree),intent(in)::self
  real(dp),intent(in)::u(:),v(:)
  if(self%kind<=9)then
   value=row_distance(u,v,self%kind,self%power,.false.,.false.,.false.)
  else
   value=extended_distance(u,v,self%kind,.false.,.false.,u/=0,v/=0)
  end if
 end function
 subroutine check_query(self,point)
  class(neighbor_ball_tree),intent(in)::self
  real(dp),intent(in)::point(:)
  call require(allocated(self%order),'Ball tree is not fitted')
  call require(size(point)==size(self%x,2),'Ball tree query feature count')
  call require(all(ieee_is_finite(point)),'Ball tree query must be finite')
 end subroutine
 subroutine tree_query(self,point,k,distances,indices)
  class(neighbor_ball_tree),intent(in)::self
  real(dp),intent(in)::point(:)
  integer,intent(in)::k
  real(dp),allocatable,intent(out)::distances(:)
  integer,allocatable,intent(out)::indices(:)
  real(dp)::lower,upper
  integer::j
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call check_query(self,point);call require(k>=1.and.k<=size(self%x,1),'Ball tree query k out of range')
  allocate(distances(k),indices(k));distances=ieee_value(0._dp,ieee_positive_inf);indices=1
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call node_bounds(self,1,point,lower,upper);call visit(1,reduced_radius(self,lower))
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
     id=self%order(at);value=point_distance(self,point,self%x(id,:))
     call neighbor_heap_push(distances,indices,value,id)
    end do
   else
    call node_bounds(self,2*node,point,left,unused);call node_bounds(self,2*node+1,point,right,unused)
    left=reduced_radius(self,left);right=reduced_radius(self,right)
    if(left<=right)then
     call visit(2*node,left);call visit(2*node+1,right)
    else
     call visit(2*node+1,right);call visit(2*node,left)
    end if
   end if
  end subroutine
 end subroutine
 subroutine tree_radius(self,point,radius,distances,indices)
  class(neighbor_ball_tree),intent(in)::self
  real(dp),intent(in)::point(:),radius
  real(dp),allocatable,intent(out)::distances(:)
  integer,allocatable,intent(out)::indices(:)
  integer::used
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call check_query(self,point)
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
   if(lo>radius)return
   if(hi<=radius)then
    do at=self%spans(node,1),self%spans(node,2)
     id=self%order(at);used=used+1;indices(used)=id
     distances(used)=actual_distance(self,point_distance(self,point,self%x(id,:)))
    end do
   else if(self%spans(node,3)==1)then
    cutoff=reduced_radius(self,radius)
    do at=self%spans(node,1),self%spans(node,2)
     id=self%order(at);value=point_distance(self,point,self%x(id,:))
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
