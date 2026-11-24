program reference_minmax_driver
 use sklearn_common,only:dp
 use sklearn_minmax_scaling,only:minmax_scaler,minmax_scale
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_positive_inf
 implicit none
 type(minmax_scaler)::model
 real(dp),allocatable::x(:,:),b(:,:),q(:,:),z(:,:),v(:,:),back(:,:),vector(:),ox(:,:),ob(:,:),oq(:,:)
 integer,allocatable::before(:),after(:)
 real(dp)::lo,hi,blo,bhi,qlo,qhi
 integer::c,cases,stream,axis,clip,qclip,mode,action,ns
 call random_seed(size=ns);allocate(before(ns),after(ns));read(*,*) cases
 do c=1,cases
  read(*,*) stream,axis,clip,qclip,mode,action,lo,hi,blo,bhi,qlo,qhi
  call read_matrix(x);call read_matrix(b);call read_matrix(q)
  model=minmax_scaler();model%feature_min=lo;model%feature_max=hi;model%clip=clip/=0
  if(mode==1) z=model%transform(q)
  if(mode==2) x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==3) q(1,1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==4) b(1,1)=ieee_value(0._dp,ieee_positive_inf)
  ox=x;ob=b;oq=q;call random_seed(get=before)
  call model%fit(x)
  if(mode==5) model%n_samples_seen_=huge(0_int64)
  if(stream/=0) then
   model%feature_min=blo;model%feature_max=bhi;call model%partial_fit(b)
  end if
  if(action==2) model%scale_=0
  if(action==3) model%scale_=-model%scale_
  if(action==4) then
   model%scale_=1.e308_dp;model%min_=-1.e308_dp
  end if
  print *,'BEGIN',c
  call emit('data_min',model%data_min_);call emit('data_max',model%data_max_);call emit('data_range',model%data_range_)
  call emit('scale',model%scale_);call emit('offset',model%min_)
  call emit('metadata',real([int(model%n_features_in_,int64),model%n_samples_seen_],dp))
  model%feature_min=qlo;model%feature_max=qhi;model%clip=qclip/=0
  z=model%transform(x);call emit('train',reshape(z,[size(z)]))
  if(action/=1) then
   v=model%transform(q);call emit('query',reshape(v,[size(v)]))
  end if
  back=model%inverse_transform(q);call emit('inverse_shape',real(shape(back),dp))
  call emit('inverse',reshape(back,[size(back)]))
  z=minmax_scale(x,lo,hi,axis);call emit('free',reshape(z,[size(z)]))
  vector=reshape(x,[size(x)]);call emit('vector',minmax_scale(vector,lo,hi,axis))
  call random_seed(get=after)
  call emit('preserved',real([merge(1,0,equal_bits(x,ox)),merge(1,0,equal_bits(b,ob)), &
    merge(1,0,equal_bits(q,oq)),merge(1,0,all(before==after))],dp))
  model%feature_min=lo;model%feature_max=hi;call model%fit(q)
  call emit('refit',real([int(model%n_features_in_,int64),model%n_samples_seen_],dp))
  call emit('refit_min',model%data_min_);call emit('refit_max',model%data_max_)
  print *,'END',c
 end do
contains
 subroutine read_matrix(a)
  real(dp),allocatable,intent(out)::a(:,:)
  integer::n,p
  read(*,*) n,p;allocate(a(n,p));read(*,*) a
 end subroutine
 logical function equal_bits(a,b)
  real(dp),intent(in)::a(:,:),b(:,:)
  equal_bits=all(transfer(a,[0_int64])==transfer(b,[0_int64]))
 end function
 subroutine emit(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:)
  print *,name,size(a);write(*,'(*(es25.16e3,1x))') a
 end subroutine
end program
