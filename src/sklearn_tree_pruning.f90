! sklearn 1.6.1 tree/_tree.pyx weakest-link cost-complexity pruning (BSD).
! Retained terms: LICENSES/scikit-learn-COPYING and THIRD_PARTY_NOTICES.md.
module sklearn_tree_pruning
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  private
  public :: tree_pruning_path,weakest_link_pruning
  type :: tree_pruning_path
    real(dp), allocatable :: ccp_alphas(:),impurities(:)
  end type
contains
  subroutine weakest_link_pruning(left,right,mass,impurity,leaves,path,alpha)
    integer, intent(in) :: left(:),right(:)
    real(dp), intent(in) :: mass(:),impurity(:)
    logical, allocatable, intent(out) :: leaves(:)
    type(tree_pruning_path), intent(out) :: path
    real(dp), optional, intent(in) :: alpha
    integer :: n,i,j,node,parent_node,chosen,top,removed,used
    integer, allocatable :: parent(:),nleaves(:),stack(:)
    logical, allocatable :: candidate(:),inside(:)
    real(dp), allocatable :: risk(:),branch(:),alphas(:),impurities(:)
    real(dp) :: effective,current,change
    n=size(left)
    call require(n>0.and.size(right)==n.and.size(mass)==n.and.size(impurity)==n,'invalid pruning node arrays')
    allocate(leaves(n),parent(n),nleaves(n),stack(n),candidate(n),inside(n),risk(n),branch(n),alphas(n),impurities(n))
    leaves=left==0;parent=0;nleaves=0;branch=0;inside=.true.;candidate=.not.leaves
    do i=1,n
      risk(i)=ieee_quotient(ieee_product(mass(i),impurity(i)),mass(1))
      if(left(i)==0) cycle
      parent(left(i))=i;parent(right(i))=i
    end do
    ! Source leaf order and ancestor accumulation preserve its rounding/ties.
    do i=1,n
      if(.not.leaves(i)) cycle
      branch(i)=risk(i);current=risk(i);node=i
      do while(node/=1)
        j=parent(node);branch(j)=ieee_difference(branch(j),-current)
        nleaves(j)=nleaves(j)+1;node=j
      end do
    end do
    used=1;alphas(used)=0;impurities(used)=branch(1)
    do while(candidate(1))
      effective=huge(1._dp);chosen=0
      do i=1,n
        if(.not.candidate(i)) cycle
        current=ieee_quotient(ieee_difference(risk(i),branch(i)),real(nleaves(i)-1,dp))
        if(ieee_is_nan(current)) cycle
        if(current<effective) then
          effective=current;chosen=i
        end if
      end do
      if(present(alpha)) then
        if(alpha<effective) exit
      end if
      ! The C source uses an uninitialized node index if no comparable link
      ! exists. Reject that undefined state instead of reading invalid memory.
      call require(chosen>0,'cost-complexity pruning has no comparable weakest link')
      top=1;stack(top)=chosen
      do while(top>0)
        node=stack(top);top=top-1
        if(.not.inside(node)) cycle
        candidate(node)=.false.;leaves(node)=.false.;inside(node)=.false.
        if(left(node)==0) cycle
        top=top+1;stack(top)=left(node);top=top+1;stack(top)=right(node)
      end do
      leaves(chosen)=.true.;inside(chosen)=.true.
      removed=nleaves(chosen)-1;nleaves(chosen)=0
      change=ieee_difference(risk(chosen),branch(chosen));branch(chosen)=risk(chosen)
      parent_node=parent(chosen)
      do while(parent_node/=0)
        nleaves(parent_node)=nleaves(parent_node)-removed
        branch(parent_node)=ieee_difference(branch(parent_node),-change)
        parent_node=parent(parent_node)
      end do
      used=used+1;alphas(used)=effective;impurities(used)=branch(1)
    end do
    path%ccp_alphas=alphas(:used);path%impurities=impurities(:used)
  end subroutine
end module
