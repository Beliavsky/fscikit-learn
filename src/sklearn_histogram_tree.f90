! Numeric best-first Newton histogram trees, sklearn 1.6.1 grower/splitting.
! See THIRD_PARTY_NOTICES.md for retained BSD-3-Clause attribution.
module sklearn_histogram_tree
  use sklearn_common, only: dp,require
  use sklearn_histogram_binning, only: histogram_bin_mapper
  use, intrinsic :: iso_fortran_env, only: int16,int64,real32
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_value,ieee_positive_inf,ieee_negative_inf
  implicit none
  private
  public :: histogram_tree,histogram_node,grow_histogram_tree
  type :: histogram_node
    integer :: feature=0,bin_threshold=0,left=0,right=0,count=0,depth=0
    real(dp) :: value=0,gain=-1,threshold=0
    logical :: missing_go_to_left=.false.
  end type
  type :: histogram_tree
    type(histogram_node), allocatable :: nodes_(:)
  contains
    procedure :: predict=>tree_predict
    procedure :: apply=>tree_apply
  end type
  type :: node_work
    integer, allocatable :: ids(:),counts(:,:)
    real(dp), allocatable :: gradients(:,:),hessians(:,:)
    real(dp) :: g=0,h=0,lower=0,upper=0,gl=0,hl=0,vl=0,vr=0
    integer :: feature=0,cut=0,nleft=0
    logical :: missing_left=.false.
  end type
contains
  real(dp) function node_value(g,h,lower,upper,l2) result(value)
    real(dp), intent(in) :: g,h,lower,upper,l2
    value=max(lower,min(upper,-g/(h+l2+1.e-15_dp)))
  end function
  subroutine histogram(work,bins,g,h,nb)
    type(node_work), intent(inout) :: work
    integer(int16), intent(in) :: bins(:,:)
    real(real32), intent(in) :: g(:),h(:)
    integer, intent(in) :: nb
    integer :: i,j,k,b
    allocate(work%gradients(nb,size(bins,2)),work%hessians(nb,size(bins,2)),work%counts(nb,size(bins,2)))
    work%gradients=0;work%hessians=0;work%counts=0
    do j=1,size(bins,2)
      do k=1,size(work%ids)
        i=work%ids(k);b=int(bins(i,j))
        work%gradients(b,j)=work%gradients(b,j)+real(g(i),dp)
        work%hessians(b,j)=work%hessians(b,j)+real(h(i),dp)
        work%counts(b,j)=work%counts(b,j)+1
      end do
    end do
  end subroutine
  subroutine evaluate(node,work,mapper,min_leaf,l2,monotonic)
    type(histogram_node), intent(inout) :: node
    type(node_work), intent(inout) :: work
    type(histogram_bin_mapper), intent(in) :: mapper
    integer, intent(in) :: min_leaf,monotonic(:)
    real(dp), intent(in) :: l2
    real(dp) :: gl,hl,gr,hr,vl,vr,gain
    integer :: j,b,nl,nr,direction,first,last,increment
    node%gain=-1
    do j=1,mapper%n_features_in_
      do direction=1,2
        if(direction==2.and..not.mapper%has_missing_values_(j)) cycle
        gl=0;hl=0;gr=0;hr=0;nl=0;nr=0
        if(direction==1) then
          first=1;last=mapper%n_bins_non_missing_(j)-1;increment=1
          if(mapper%has_missing_values_(j)) last=last+1
        else
          first=mapper%n_bins_non_missing_(j)-1;last=1;increment=-1
        end if
        do b=first,last,increment
          if(direction==1) then
            gl=gl+work%gradients(b,j);hl=hl+work%hessians(b,j);nl=nl+work%counts(b,j)
            gr=work%g-gl;hr=work%h-hl;nr=node%count-nl
          else
            gr=gr+work%gradients(b+1,j);hr=hr+work%hessians(b+1,j);nr=nr+work%counts(b+1,j)
            gl=work%g-gr;hl=work%h-hr;nl=node%count-nr
          end if
          if(nl<min_leaf.or.nr<min_leaf.or.hl<1.e-3_dp.or.hr<1.e-3_dp) cycle
          vl=node_value(gl,hl,work%lower,work%upper,l2)
          vr=node_value(gr,hr,work%lower,work%upper,l2)
          if(monotonic(j)==1.and.vl>vr.or.monotonic(j)==-1.and.vl<vr) cycle
          gain=work%g*node%value-gl*vl-gr*vr
          if(gain<=0.or.gain<=node%gain) cycle
          node%gain=gain;work%feature=j;work%cut=b;work%missing_left=direction==2
          work%gl=gl;work%hl=hl;work%nleft=nl;work%vl=vl;work%vr=vr
        end do
      end do
    end do
  end subroutine
  subroutine heap_push(heap,n,id,nodes)
    integer, intent(inout) :: heap(:),n
    integer, intent(in) :: id
    type(histogram_node), intent(in) :: nodes(:)
    integer :: pos,parent
    n=n+1;pos=n
    do while(pos>1)
      parent=pos/2
      if(nodes(id)%gain<=nodes(heap(parent))%gain) exit
      heap(pos)=heap(parent);pos=parent
    end do
    heap(pos)=id
  end subroutine
  integer function heap_pop(heap,n,nodes) result(id)
    integer, intent(inout) :: heap(:),n
    type(histogram_node), intent(in) :: nodes(:)
    integer :: last,pos,child,parent
    id=heap(1);last=heap(n);n=n-1
    if(n==0) return
    pos=1;child=2
    do while(child<=n)
      if(child<n) then
        if(.not.(nodes(heap(child))%gain>nodes(heap(child+1))%gain)) child=child+1
      end if
      heap(pos)=heap(child);pos=child;child=2*pos
    end do
    do while(pos>1)
      parent=pos/2
      if(nodes(last)%gain<=nodes(heap(parent))%gain) exit
      heap(pos)=heap(parent);pos=parent
    end do
    heap(pos)=last
  end function
  function grow_histogram_tree(bins,g,h,mapper,min_leaf,max_leaves,max_depth,l2,rate,monotonic) result(tree)
    integer(int16), intent(in) :: bins(:,:)
    real(real32), intent(in) :: g(:),h(:)
    type(histogram_bin_mapper), intent(in) :: mapper
    integer, intent(in) :: min_leaf,max_leaves,max_depth,monotonic(:)
    real(dp), intent(in) :: l2,rate
    type(histogram_tree) :: tree
    type(histogram_node), allocatable :: nodes(:)
    type(node_work), allocatable :: work(:)
    integer, allocatable :: heap(:),map(:)
    logical, allocatable :: left_mask(:)
    integer :: n,limit,capacity,used,leaves,nheap,id,left,right,small,large,j,k,cut,f,position
    real(dp) :: middle
    logical :: can_left,can_right
    n=size(bins,1);limit=max(1,n/min_leaf)
    if(max_leaves>0) limit=min(limit,max_leaves)
    call require(int(limit,int64)*2<huge(1),'histogram tree exceeds integer capacity')
    capacity=2*limit-1;allocate(nodes(capacity),work(capacity),heap(capacity),map(capacity))
    nodes(1)%count=n;work(1)%ids=[(j,j=1,n)]
    work(1)%g=sum(real(g,dp));work(1)%h=sum(real(h,dp))
    work(1)%lower=ieee_value(0._dp,ieee_negative_inf);work(1)%upper=ieee_value(0._dp,ieee_positive_inf)
    used=1;leaves=1;nheap=0
    if(int(n,int64)>=2_int64*min_leaf.and.work(1)%h>=1.e-3_dp.and.limit>1) then
      call histogram(work(1),bins,g,h,mapper%missing_bin_)
      call evaluate(nodes(1),work(1),mapper,min_leaf,l2,monotonic)
      if(nodes(1)%gain>0) call heap_push(heap,nheap,1,nodes)
    end if
    do while(nheap>0.and.leaves<limit)
      id=heap_pop(heap,nheap,nodes);left=used+1;right=used+2;used=right;leaves=leaves+1
      f=work(id)%feature;cut=work(id)%cut
      nodes(id)%feature=f;nodes(id)%bin_threshold=cut;nodes(id)%left=left;nodes(id)%right=right
      nodes(id)%missing_go_to_left=work(id)%missing_left
      if(.not.mapper%has_missing_values_(f)) &
        nodes(id)%missing_go_to_left=work(id)%nleft>nodes(id)%count-work(id)%nleft
      nodes(id)%threshold=ieee_value(0._dp,ieee_positive_inf)
      if(cut<mapper%n_bins_non_missing_(f)) nodes(id)%threshold=mapper%bin_thresholds_(f)%values(cut)
      if(allocated(left_mask)) deallocate(left_mask)
      allocate(left_mask(size(work(id)%ids)))
      do j=1,size(left_mask)
        k=int(bins(work(id)%ids(j),f))
        left_mask(j)=k<=cut
        if(k==mapper%missing_bin_) left_mask(j)=nodes(id)%missing_go_to_left
      end do
      work(left)%ids=pack(work(id)%ids,left_mask);work(right)%ids=pack(work(id)%ids,.not.left_mask)
      nodes(left)%count=size(work(left)%ids);nodes(right)%count=size(work(right)%ids)
      nodes(left)%depth=nodes(id)%depth+1;nodes(right)%depth=nodes(left)%depth
      nodes(left)%value=work(id)%vl;nodes(right)%value=work(id)%vr
      work(left)%g=work(id)%gl;work(left)%h=work(id)%hl
      work(right)%g=work(id)%g-work(id)%gl;work(right)%h=work(id)%h-work(id)%hl
      work(left)%lower=work(id)%lower;work(left)%upper=work(id)%upper
      work(right)%lower=work(id)%lower;work(right)%upper=work(id)%upper
      middle=(nodes(left)%value+nodes(right)%value)*.5_dp
      if(monotonic(f)==1) then
        work(left)%upper=middle;work(right)%lower=middle
      else if(monotonic(f)==-1) then
        work(left)%lower=middle;work(right)%upper=middle
      end if
      can_left=int(nodes(left)%count,int64)>=2_int64*min_leaf
      can_right=int(nodes(right)%count,int64)>=2_int64*min_leaf
      if(max_depth>0) then
        can_left=can_left.and.nodes(left)%depth<max_depth
        can_right=can_right.and.nodes(right)%depth<max_depth
      end if
      if(leaves==limit) then
        can_left=.false.;can_right=.false.
      end if
      if(can_left.or.can_right) then
        small=right;large=left
        if(nodes(left)%count<nodes(right)%count) then
          small=left;large=right
        end if
        call histogram(work(small),bins,g,h,mapper%missing_bin_)
        work(large)%gradients=work(id)%gradients-work(small)%gradients
        work(large)%hessians=work(id)%hessians-work(small)%hessians
        work(large)%counts=work(id)%counts-work(small)%counts
        if(can_left) then
          call evaluate(nodes(left),work(left),mapper,min_leaf,l2,monotonic)
          if(nodes(left)%gain>0) call heap_push(heap,nheap,left,nodes)
        end if
        if(can_right) then
          call evaluate(nodes(right),work(right),mapper,min_leaf,l2,monotonic)
          if(nodes(right)%gain>0) call heap_push(heap,nheap,right,nodes)
        end if
      end if
      deallocate(work(id)%ids,work(id)%gradients,work(id)%hessians,work(id)%counts)
    end do
    allocate(tree%nodes_(used));position=0;map=0
    call flatten(1)
  contains
    recursive subroutine flatten(old)
      integer, intent(in) :: old
      integer :: current
      position=position+1;current=position;map(old)=current;tree%nodes_(current)=nodes(old)
      if(nodes(old)%left==0) then
        tree%nodes_(current)%value=nodes(old)%value*rate
      else
        call flatten(nodes(old)%left);tree%nodes_(current)%left=map(nodes(old)%left)
        call flatten(nodes(old)%right);tree%nodes_(current)%right=map(nodes(old)%right)
      end if
    end subroutine
  end function
  function tree_apply(self,x) result(ids)
    class(histogram_tree), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: ids(size(x,1)),i,node,f
    logical :: go_left
    call require(allocated(self%nodes_),'histogram tree is not fitted')
    do i=1,size(x,1)
      node=1
      do while(self%nodes_(node)%left/=0)
        f=self%nodes_(node)%feature
        if(ieee_is_nan(x(i,f))) then
          go_left=self%nodes_(node)%missing_go_to_left
        else
          go_left=.false.
          if(.not.ieee_is_nan(self%nodes_(node)%threshold)) go_left=x(i,f)<=self%nodes_(node)%threshold
        end if
        if(go_left) then
          node=self%nodes_(node)%left
        else
          node=self%nodes_(node)%right
        end if
      end do
      ids(i)=node
    end do
  end function
  function tree_predict(self,x) result(prediction)
    class(histogram_tree), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: prediction(size(x,1))
    integer :: ids(size(x,1)),i
    ids=self%apply(x)
    do i=1,size(ids)
      prediction(i)=self%nodes_(ids(i))%value
    end do
  end function
end module
