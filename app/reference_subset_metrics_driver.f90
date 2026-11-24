program reference_subset_metrics_driver
 use sklearn_subset_metrics,only:accuracy_score,zero_one_loss
 use sklearn_sparse,only:csr_matrix
 use sklearn_common,only:dp
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type::data_holder
  integer,allocatable::i(:),ii(:,:)
  real(dp),allocatable::d(:),dd(:,:)
  logical,allocatable::l(:),ll(:,:)
  type(csr_matrix)::s
 end type
 type(data_holder)::a,b,saved_a,saved_b
 real(dp),allocatable,target::storage(:)
 real(dp),pointer::weights(:)=>null()
 real(dp),allocatable::saved_weights(:)
 integer::ny,py,ky,np,pp,kp,weighted,nw,norm
 read(*,*) ny,py,ky,np,pp,kp,weighted,nw,norm
 call read_data(a,ny,py,ky);call read_data(b,np,pp,kp)
 if(weighted>0) then
  allocate(storage(max(1,2*nw)))
  select case(weighted)
  case(1);weights=>storage(:nw)
  case(2);weights=>storage(1:2*nw:2)
  case(3);weights=>storage(nw:1:-1)
  end select
  read(*,*) weights;saved_weights=weights
 end if
 saved_a=a;saved_b=b
 select case(ky)
 case(0);call predict_i(a%i)
 case(1);call predict_r(a%d)
 case(2);call predict_l(a%l)
 case(3);call predict_i(a%ii)
 case(4);call predict_r(a%dd)
 case(5);call predict_l(a%ll)
 case(6);call predict_c(a%s)
 end select
 print *, 'identity',merge(1,0,same(a,saved_a,ky).and.same(b,saved_b,kp).and.weight_identity())
contains
 subroutine read_data(v,n,p,kind)
  type(data_holder),intent(out)::v
  integer,intent(in)::n,p,kind
  integer,allocatable::iv(:),im(:,:)
  integer::nnz
  select case(kind)
  case(0);allocate(v%i(n));read(*,*) v%i
  case(1);allocate(v%d(n));read(*,*) v%d
  case(2);allocate(iv(n));read(*,*) iv;v%l=iv/=0
  case(3);allocate(v%ii(n,p));read(*,*) v%ii
  case(4);allocate(v%dd(n,p));read(*,*) v%dd
  case(5);allocate(im(n,p));read(*,*) im;v%ll=im/=0
  case(6)
   read(*,*) nnz;v%s%n_rows=n;v%s%n_cols=p
   allocate(v%s%indptr(n+1),v%s%indices(nnz),v%s%data(nnz))
   read(*,*) v%s%indptr
   read(*,*) v%s%indices
   read(*,*) v%s%data
  end select
 end subroutine
 subroutine predict_i(y)
  integer,intent(in)::y(..)
  real(dp)::accuracy,loss
  select rank(y)
  rank(1)
  select case(kp)
  case(0)
   accuracy=accuracy_score(y,b%i,norm==1,weights)
   loss=zero_one_loss(y,b%i,norm==1,weights)
  case(1)
   accuracy=accuracy_score(y,b%d,norm==1,weights)
   loss=zero_one_loss(y,b%d,norm==1,weights)
  case(2)
   accuracy=accuracy_score(y,b%l,norm==1,weights)
   loss=zero_one_loss(y,b%l,norm==1,weights)
  case(3)
   accuracy=accuracy_score(y,b%ii,norm==1,weights)
   loss=zero_one_loss(y,b%ii,norm==1,weights)
  case(4)
   accuracy=accuracy_score(y,b%dd,norm==1,weights)
   loss=zero_one_loss(y,b%dd,norm==1,weights)
  case(5)
   accuracy=accuracy_score(y,b%ll,norm==1,weights)
   loss=zero_one_loss(y,b%ll,norm==1,weights)
  case(6)
   accuracy=accuracy_score(y,b%s,norm==1,weights)
   loss=zero_one_loss(y,b%s,norm==1,weights)
  end select
  rank(2)
  select case(kp)
  case(0)
   accuracy=accuracy_score(y,b%i,norm==1,weights)
   loss=zero_one_loss(y,b%i,norm==1,weights)
  case(1)
   accuracy=accuracy_score(y,b%d,norm==1,weights)
   loss=zero_one_loss(y,b%d,norm==1,weights)
  case(2)
   accuracy=accuracy_score(y,b%l,norm==1,weights)
   loss=zero_one_loss(y,b%l,norm==1,weights)
  case(3)
   accuracy=accuracy_score(y,b%ii,norm==1,weights)
   loss=zero_one_loss(y,b%ii,norm==1,weights)
  case(4)
   accuracy=accuracy_score(y,b%dd,norm==1,weights)
   loss=zero_one_loss(y,b%dd,norm==1,weights)
  case(5)
   accuracy=accuracy_score(y,b%ll,norm==1,weights)
   loss=zero_one_loss(y,b%ll,norm==1,weights)
  case(6)
   accuracy=accuracy_score(y,b%s,norm==1,weights)
   loss=zero_one_loss(y,b%s,norm==1,weights)
  end select
  end select
  print *, 'accuracy',accuracy
  print *, 'loss',loss
 end subroutine
 subroutine predict_r(y)
  real(dp),intent(in)::y(..)
  real(dp)::accuracy,loss
  select rank(y)
  rank(1)
  select case(kp)
  case(0)
   accuracy=accuracy_score(y,b%i,norm==1,weights)
   loss=zero_one_loss(y,b%i,norm==1,weights)
  case(1)
   accuracy=accuracy_score(y,b%d,norm==1,weights)
   loss=zero_one_loss(y,b%d,norm==1,weights)
  case(2)
   accuracy=accuracy_score(y,b%l,norm==1,weights)
   loss=zero_one_loss(y,b%l,norm==1,weights)
  case(3)
   accuracy=accuracy_score(y,b%ii,norm==1,weights)
   loss=zero_one_loss(y,b%ii,norm==1,weights)
  case(4)
   accuracy=accuracy_score(y,b%dd,norm==1,weights)
   loss=zero_one_loss(y,b%dd,norm==1,weights)
  case(5)
   accuracy=accuracy_score(y,b%ll,norm==1,weights)
   loss=zero_one_loss(y,b%ll,norm==1,weights)
  case(6)
   accuracy=accuracy_score(y,b%s,norm==1,weights)
   loss=zero_one_loss(y,b%s,norm==1,weights)
  end select
  rank(2)
  select case(kp)
  case(0)
   accuracy=accuracy_score(y,b%i,norm==1,weights)
   loss=zero_one_loss(y,b%i,norm==1,weights)
  case(1)
   accuracy=accuracy_score(y,b%d,norm==1,weights)
   loss=zero_one_loss(y,b%d,norm==1,weights)
  case(2)
   accuracy=accuracy_score(y,b%l,norm==1,weights)
   loss=zero_one_loss(y,b%l,norm==1,weights)
  case(3)
   accuracy=accuracy_score(y,b%ii,norm==1,weights)
   loss=zero_one_loss(y,b%ii,norm==1,weights)
  case(4)
   accuracy=accuracy_score(y,b%dd,norm==1,weights)
   loss=zero_one_loss(y,b%dd,norm==1,weights)
  case(5)
   accuracy=accuracy_score(y,b%ll,norm==1,weights)
   loss=zero_one_loss(y,b%ll,norm==1,weights)
  case(6)
   accuracy=accuracy_score(y,b%s,norm==1,weights)
   loss=zero_one_loss(y,b%s,norm==1,weights)
  end select
  end select
  print *, 'accuracy',accuracy
  print *, 'loss',loss
 end subroutine
 subroutine predict_l(y)
  logical,intent(in)::y(..)
  real(dp)::accuracy,loss
  select rank(y)
  rank(1)
  select case(kp)
  case(0)
   accuracy=accuracy_score(y,b%i,norm==1,weights)
   loss=zero_one_loss(y,b%i,norm==1,weights)
  case(1)
   accuracy=accuracy_score(y,b%d,norm==1,weights)
   loss=zero_one_loss(y,b%d,norm==1,weights)
  case(2)
   accuracy=accuracy_score(y,b%l,norm==1,weights)
   loss=zero_one_loss(y,b%l,norm==1,weights)
  case(3)
   accuracy=accuracy_score(y,b%ii,norm==1,weights)
   loss=zero_one_loss(y,b%ii,norm==1,weights)
  case(4)
   accuracy=accuracy_score(y,b%dd,norm==1,weights)
   loss=zero_one_loss(y,b%dd,norm==1,weights)
  case(5)
   accuracy=accuracy_score(y,b%ll,norm==1,weights)
   loss=zero_one_loss(y,b%ll,norm==1,weights)
  case(6)
   accuracy=accuracy_score(y,b%s,norm==1,weights)
   loss=zero_one_loss(y,b%s,norm==1,weights)
  end select
  rank(2)
  select case(kp)
  case(0)
   accuracy=accuracy_score(y,b%i,norm==1,weights)
   loss=zero_one_loss(y,b%i,norm==1,weights)
  case(1)
   accuracy=accuracy_score(y,b%d,norm==1,weights)
   loss=zero_one_loss(y,b%d,norm==1,weights)
  case(2)
   accuracy=accuracy_score(y,b%l,norm==1,weights)
   loss=zero_one_loss(y,b%l,norm==1,weights)
  case(3)
   accuracy=accuracy_score(y,b%ii,norm==1,weights)
   loss=zero_one_loss(y,b%ii,norm==1,weights)
  case(4)
   accuracy=accuracy_score(y,b%dd,norm==1,weights)
   loss=zero_one_loss(y,b%dd,norm==1,weights)
  case(5)
   accuracy=accuracy_score(y,b%ll,norm==1,weights)
   loss=zero_one_loss(y,b%ll,norm==1,weights)
  case(6)
   accuracy=accuracy_score(y,b%s,norm==1,weights)
   loss=zero_one_loss(y,b%s,norm==1,weights)
  end select
  end select
  print *, 'accuracy',accuracy
  print *, 'loss',loss
 end subroutine
 subroutine predict_c(y)
  type(csr_matrix),intent(in)::y
  real(dp)::accuracy,loss
  select case(kp)
  case(0)
   accuracy=accuracy_score(y,b%i,norm==1,weights)
   loss=zero_one_loss(y,b%i,norm==1,weights)
  case(1)
   accuracy=accuracy_score(y,b%d,norm==1,weights)
   loss=zero_one_loss(y,b%d,norm==1,weights)
  case(2)
   accuracy=accuracy_score(y,b%l,norm==1,weights)
   loss=zero_one_loss(y,b%l,norm==1,weights)
  case(3)
   accuracy=accuracy_score(y,b%ii,norm==1,weights)
   loss=zero_one_loss(y,b%ii,norm==1,weights)
  case(4)
   accuracy=accuracy_score(y,b%dd,norm==1,weights)
   loss=zero_one_loss(y,b%dd,norm==1,weights)
  case(5)
   accuracy=accuracy_score(y,b%ll,norm==1,weights)
   loss=zero_one_loss(y,b%ll,norm==1,weights)
  case(6)
   accuracy=accuracy_score(y,b%s,norm==1,weights)
   loss=zero_one_loss(y,b%s,norm==1,weights)
  end select
  print *, 'accuracy',accuracy
  print *, 'loss',loss
 end subroutine
 logical function weight_identity() result(same_values)
  same_values=.true.
  if(weighted>0) same_values=all(transfer(weights,[0_int64],size(weights))== &
   transfer(saved_weights,[0_int64],size(saved_weights)))
 end function
 logical function same(x,y,k) result(equal)
  type(data_holder),intent(in)::x,y
  integer,intent(in)::k
  equal=.false.
  select case(k)
  case(0);equal=all(x%i==y%i)
  case(1);equal=all(transfer(x%d,[0_int64],size(x%d))==transfer(y%d,[0_int64],size(y%d)))
  case(2);equal=all(x%l.eqv.y%l)
  case(3);equal=all(x%ii==y%ii)
  case(4);equal=all(transfer(x%dd,[0_int64],size(x%dd))==transfer(y%dd,[0_int64],size(y%dd)))
  case(5);equal=all(x%ll.eqv.y%ll)
  case(6)
   equal=x%s%n_rows==y%s%n_rows.and.x%s%n_cols==y%s%n_cols.and. &
    all(x%s%indptr==y%s%indptr).and.all(x%s%indices==y%s%indices).and. &
    all(transfer(x%s%data,[0_int64],size(x%s%data))==transfer(y%s%data,[0_int64],size(y%s%data)))
  end select
 end function
end program
