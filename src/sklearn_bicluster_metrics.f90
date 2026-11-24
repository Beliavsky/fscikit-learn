! sklearn 1.6.1 metrics/cluster/_bicluster.py contracts; retained BSD attribution.
! Native rectangular shortest-augmenting-path assignment is an MIT contribution.
module sklearn_bicluster_metrics
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_positive_inf
  implicit none
  private
  public :: consensus_score,bicluster_similarity
  abstract interface
    function bicluster_similarity(a_rows,a_columns,b_rows,b_columns) result(value)
      import dp
      real(dp), intent(in) :: a_rows(:),a_columns(:),b_rows(:),b_columns(:)
      real(dp) :: value
    end function
  end interface
contains
  function indicator_array(x) result(y)
    class(*), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call require(size(x,1)>0.and.size(x,2)>0,'consensus indicators must have nonempty axes')
    select type(x)
    type is(logical)
      y=merge(1._dp,0._dp,x)
    type is(integer)
      y=real(x,dp)
    type is(integer(int64))
      y=real(x,dp)
    type is(real(dp))
      y=x
    class default
      call require(.false.,'consensus indicators need logical, integer, int64 or real64 values')
    end select
    call require(all(ieee_is_finite(y)),'consensus indicators must be finite')
  end function

  real(dp) function intersection(a,b) result(value)
    real(dp), intent(in) :: a(:),b(:)
    real(dp), allocatable :: products(:)
    integer :: i,n,ia,ib
    call require(size(a)==size(b).or.size(a)==1.or.size(b)==1,'consensus indicator widths cannot broadcast')
    n=max(size(a),size(b));allocate(products(n))
    do i=1,n
      ia=min(i,size(a));ib=min(i,size(b))
      products(i)=ieee_product(a(ia),b(ib))
    end do
    value=ieee_total(products)
  end function

  real(dp) function jaccard(a_rows,a_columns,b_rows,b_columns) result(value)
    real(dp), intent(in) :: a_rows(:),a_columns(:),b_rows(:),b_columns(:)
    real(dp) :: overlap,a_size,b_size,denominator
    overlap=ieee_product(intersection(a_rows,b_rows),intersection(a_columns,b_columns))
    a_size=ieee_product(ieee_total(a_rows),ieee_total(a_columns))
    b_size=ieee_product(ieee_total(b_rows),ieee_total(b_columns))
    denominator=ieee_difference(ieee_difference(a_size,-b_size),overlap)
    value=ieee_quotient(overlap,denominator)
  end function

  function minimum_assignment(cost) result(columns)
    ! Rectangular Hungarian potentials: one column per row, n <= m.
    real(dp), intent(in) :: cost(:,:)
    integer :: columns(size(cost,1))
    real(dp) :: u(0:size(cost,1)),v(0:size(cost,2)),best(size(cost,2)),delta,current,infinity
    integer :: owner(0:size(cost,2)),previous(size(cost,2)),i,j,j0,j1,i0,n,m
    logical :: used(0:size(cost,2))
    n=size(cost,1);m=size(cost,2);u=0;v=0;owner=0
    infinity=ieee_value(0._dp,ieee_positive_inf)
    do i=1,n
      owner(0)=i;j0=0;best=infinity;used=.false.;previous=0
      do
        used(j0)=.true.;i0=owner(j0);delta=infinity;j1=0
        do j=1,m
          if(used(j)) cycle
          current=ieee_difference(ieee_difference(cost(i0,j),u(i0)),v(j))
          if(current<best(j)) then
            best(j)=current;previous(j)=j0
          end if
          if(best(j)<delta) then
            delta=best(j);j1=j
          end if
        end do
        call require(ieee_is_finite(delta).and.j1>0,'consensus assignment is infeasible')
        do j=0,m
          if(used(j)) then
            u(owner(j))=ieee_difference(u(owner(j)),-delta)
            v(j)=ieee_difference(v(j),delta)
          else if(j>0) then
            best(j)=ieee_difference(best(j),delta)
          end if
        end do
        j0=j1
        if(owner(j0)==0) exit
      end do
      do
        j1=previous(j0);owner(j0)=owner(j1);j0=j1
        if(j0==0) exit
      end do
    end do
    columns=0
    do j=1,m
      if(owner(j)>0) columns(owner(j))=j
    end do
  end function

  real(dp) function consensus_score(a_rows,a_columns,b_rows,b_columns,similarity) result(value)
    class(*), intent(in) :: a_rows(:,:),a_columns(:,:),b_rows(:,:),b_columns(:,:)
    procedure(bicluster_similarity), optional :: similarity
    real(dp), allocatable :: ar(:,:),ac(:,:),br(:,:),bc(:,:),scores(:,:),cost(:,:),selected(:)
    integer, allocatable :: columns(:)
    integer :: i,j,na,nb
    call require(size(a_rows,1)==size(a_columns,1),'consensus first row/column bicluster counts differ')
    call require(size(b_rows,1)==size(b_columns,1),'consensus second row/column bicluster counts differ')
    ar=indicator_array(a_rows);ac=indicator_array(a_columns)
    br=indicator_array(b_rows);bc=indicator_array(b_columns)
    na=size(ar,1);nb=size(br,1);allocate(scores(na,nb),cost(na,nb))
    do i=1,na
      do j=1,nb
        if(present(similarity)) then
          scores(i,j)=similarity(ar(i,:),ac(i,:),br(j,:),bc(j,:))
        else
          scores(i,j)=jaccard(ar(i,:),ac(i,:),br(j,:),bc(j,:))
        end if
        cost(i,j)=ieee_difference(1._dp,scores(i,j))
        call require(.not.ieee_is_nan(cost(i,j)),'consensus similarity produced an invalid assignment cost')
        call require(cost(i,j)>=0.or.ieee_is_finite(cost(i,j)), &
          'consensus similarity produced an invalid assignment cost')
      end do
    end do
    allocate(selected(min(na,nb)))
    if(na<=nb) then
      columns=minimum_assignment(cost)
      do i=1,na
        selected(i)=scores(i,columns(i))
      end do
    else
      columns=minimum_assignment(transpose(cost))
      do i=1,nb
        selected(i)=scores(columns(i),i)
      end do
    end if
    value=ieee_quotient(ieee_total(selected),real(max(na,nb),dp))
  end function
end module
